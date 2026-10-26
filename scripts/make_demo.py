from pathlib import Path
import argparse, json
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='data/demo')
    a = p.parse_args()
    folder = Path(a.output)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(999)
    n_users, n_items = (8, 16)
    lines = ['userID\titemID\tx_label']
    for user in range(n_users):
        items = rng.choice(n_items, size=5, replace=False)
        lines += [f'{user}\t{i}\t{(0 if at < 3 else at - 2)}' for at, i in enumerate(items)]
    (folder / 'demo.inter').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (folder / 'items.tsv').write_text('item_id\titemID\n' + ''.join((f'item-{i}\t{i}\n' for i in range(n_items))), encoding='utf-8')
    for name in ('image', 'text', 'knowledge'):
        np.save(folder / (name + '.npy'), rng.standard_normal((n_items, 8)).astype(np.float32))
    (folder / 'dataset.json').write_text(json.dumps({'schema': 'frieren.dataset.v1', 'name': 'demo', 'n_users': n_users, 'n_items': n_items, 'interaction_file': 'demo.inter', 'item_mapping_file': 'items.tsv', 'image_features': 'image.npy', 'text_features': 'text.npy', 'knowledge_embeddings': 'knowledge.npy'}, indent=2), encoding='utf-8')
    print(folder / 'dataset.json')
if __name__ == '__main__':
    main()
