from pathlib import Path

train_dir = Path('F:/EDTH/acoustic-drone-detector/data/combined_dataset/train')

print('\n=== Updated Training Dataset ===')
for label in ['background', 'drone', 'helicopter']:
    count = len(list((train_dir / label).glob('*.wav')))
    print(f'{label:12s}: {count:4d} samples')

total = sum(len(list((train_dir / label).glob('*.wav'))) for label in ['background', 'drone', 'helicopter'])
print(f'{"TOTAL":12s}: {total:4d} samples')
