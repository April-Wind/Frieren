from pathlib import Path
from dataclasses import replace
import argparse
import json
import numpy as np
from .contracts import GraphArtifact
from .data import load_dataset, dataset_spec, load_content_features
from .config import make_dataclass_config
from .io import atomic_json, file_sha256

def _require_order(graph, data):
    if graph.item_ids != data.item_ids:
        raise ValueError('Graph and dataset use different item mappings/order')

def _print_artifact(graph, output):
    graph.save(output)
    print(json.dumps({'graph': str(output), 'items': graph.n_items, 'nnz': graph.operator._nnz(), 'metadata': graph.metadata}, ensure_ascii=False))

def build_reference(args):
    from .estimators.knowledge import KnowledgeEstimator
    import torch
    data = load_dataset(args.dataset)
    spec = dataset_spec(args.dataset)
    views = []
    for key in ('image_features', 'text_features'):
        if spec.get(key):
            a = np.load(spec[key], allow_pickle=False)
            views.append((key, KnowledgeEstimator(args.k, include_self=True).build(a, data.item_ids).operator))
    if not views:
        raise ValueError('Reference graph requires image or text features')
    if len(views) == 2:
        operator = (args.image_weight * views[0][1] + (1 - args.image_weight) * views[1][1]).coalesce()
    else:
        operator = views[0][1]
    graph = GraphArtifact(operator, data.item_ids, {'estimator': 'multimodal_reference', 'recipe': 'feature_knn_weighted_sum', 'k': args.k, 'include_self': True, 'image_weight': args.image_weight, 'orientation': 'receiver_sender', 'input_hashes': {name: file_sha256(spec[name]) for name, _ in views}})
    _print_artifact(graph, args.output)

def build_knowledge(args):
    from .estimators.knowledge import KnowledgeEstimator, align_entity_embeddings
    data = load_dataset(args.dataset)
    vectors = np.load(args.embeddings, allow_pickle=False)
    if bool(args.entity_ids) != bool(args.item_to_entity):
        raise ValueError('Provide both --entity-ids and --item-to-entity for entity-level vectors')
    if args.entity_ids:
        entities = json.loads(Path(args.entity_ids).read_text(encoding='utf-8'))
        mapping = json.loads(Path(args.item_to_entity).read_text(encoding='utf-8'))
        vectors = align_entity_embeddings(vectors, entities, mapping, data.item_ids)
    graph = KnowledgeEstimator(args.k, include_self=not args.exclude_self).build(vectors, data.item_ids)
    graph.metadata['embedding_sha256'] = file_sha256(args.embeddings)
    if args.entity_ids:
        graph.metadata['entity_ids_sha256'] = file_sha256(args.entity_ids)
        graph.metadata['item_to_entity_sha256'] = file_sha256(args.item_to_entity)
    _print_artifact(graph, args.output)

def build_behavior(args):
    from .estimators.behavior import BehaviorEstimator
    data = load_dataset(args.dataset)
    fallback = GraphArtifact.load(args.fallback)
    _require_order(fallback, data)
    estimator = BehaviorEstimator(regularization=args.regularization, k=args.k).fit(data.train)
    graph = estimator.build(data.item_ids, fallback=fallback)
    graph.metadata['training_interaction_sha256'] = data.metadata['interaction_sha256']
    _print_artifact(graph, args.output)
    if args.coefficients:
        Path(args.coefficients).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.coefficients, estimator.coefficients_)

def refine(args):
    from .weighting import refine_weights
    data = load_dataset(args.dataset)
    graph = GraphArtifact.load(args.graph)
    _require_order(graph, data)
    output = refine_weights(graph, data.train, alpha=args.alpha)
    output.metadata['training_interaction_sha256'] = data.metadata['interaction_sha256']
    _print_artifact(output, args.output)

def train_estimator(args):
    from .estimators.learnable import LearnableEstimatorConfig, train_learnable
    data = load_dataset(args.dataset)
    config = make_dataclass_config(LearnableEstimatorConfig, args.config)
    features = None if config.behavior_only else np.load(args.features, allow_pickle=False) if args.features else load_content_features(args.dataset, data.n_items)
    result = train_learnable(data, features, config, args.output_dir, device=args.device, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, default=str))

def export_estimator(args):
    from .estimators.learnable import export_learned_graph
    data = load_dataset(args.dataset)
    features = None if args.behavior_only else np.load(args.features, allow_pickle=False) if args.features else load_content_features(args.dataset, data.n_items)
    graph = export_learned_graph(data, features, args.checkpoint, k=args.k, device=args.device)
    _print_artifact(graph, args.output)

def train_backbone(args):
    from .training import TrainConfig, train_freedom
    data = load_dataset(args.dataset)
    graph = GraphArtifact.load(args.graph)
    _require_order(graph, data)
    config = make_dataclass_config(TrainConfig, args.config)
    if args.patience is not None:
        config = replace(config, patience=args.patience)
    result = train_freedom(data, graph, config, args.output_dir, device=args.device, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, default=str))

def adapt(args):
    from .adapters import adapt_graph
    graph = GraphArtifact.load(args.graph)
    _print_artifact(adapt_graph(graph, args.backbone), args.output)

def replay(args):
    from .training import replay_freedom
    data = load_dataset(args.dataset)
    graph_paths = json.loads(Path(args.graphs).read_text(encoding='utf-8'))
    root = Path(args.graphs).resolve().parent
    graphs = {name: GraphArtifact.load(root / path) for name, path in graph_paths.items()}
    for g in graphs.values():
        _require_order(g, data)
    result = replay_freedom(data, graphs, args.checkpoint, device=args.device)
    atomic_json(args.output, result)
    print(json.dumps({'output': str(args.output), 'graphs': list(graphs)}))

def main(argv=None):
    parser = argparse.ArgumentParser(description='Frieren: multi-source item relation estimation')
    commands = parser.add_subparsers(dest='command', required=True)

    def command(name, handler, dataset=True):
        p = commands.add_parser(name)
        p.set_defaults(handler=handler)
        if dataset:
            p.add_argument('--dataset', required=True, help='Fixed-split dataset JSON manifest')
        return p
    p = command('build-reference', build_reference)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--image-weight', type=float, default=0.1)
    p.add_argument('--output', required=True)
    p = command('build-k', build_knowledge)
    p.add_argument('--embeddings', required=True)
    p.add_argument('--entity-ids')
    p.add_argument('--item-to-entity')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--exclude-self', action='store_true')
    p.add_argument('--output', required=True)
    p = command('build-b', build_behavior)
    p.add_argument('--fallback', required=True)
    p.add_argument('--regularization', type=float, default=100)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--coefficients')
    p.add_argument('--output', required=True)
    p = command('refine-bgnw', refine)
    p.add_argument('--graph', required=True)
    p.add_argument('--alpha', type=float, default=1)
    p.add_argument('--output', required=True)
    for name, fn in (('train-l', train_estimator), ('train-freedom', train_backbone)):
        p = command(name, fn)
        p.add_argument('--config')
        p.add_argument('--output-dir', required=True)
        p.add_argument('--device', default='cpu')
        p.add_argument('--resume', action='store_true')
        if name == 'train-freedom':
            p.add_argument('--graph', required=True)
            p.add_argument('--patience', type=int, help='Stop after this many non-improving validation Recall@20 epochs')
        else:
            p.add_argument('--features', help='Optional already-prepared item-aligned fixed content matrix')
    p = command('export-l', export_estimator)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', required=True)
    p.add_argument('--features')
    p.add_argument('--behavior-only', action='store_true')
    p = command('adapt', adapt, dataset=False)
    p.add_argument('--graph', required=True)
    p.add_argument('--backbone', choices=['FREEDOM', 'MENTOR', 'SMORE', 'MSCA'], required=True)
    p.add_argument('--output', required=True)
    p = command('replay', replay)
    p.add_argument('--graphs', required=True, help='JSON name→graph path, relative to this JSON')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'build-reference' and (not 0 <= args.image_weight <= 1):
        parser.error('image weight must lie in [0,1]')
    args.handler(args)
