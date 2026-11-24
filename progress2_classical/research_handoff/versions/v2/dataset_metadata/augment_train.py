import argparse
import csv
import json
import random
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

parser = argparse.ArgumentParser()
parser.add_argument('--per-class', type=int, default=75)
parser.add_argument('--allow-review', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parent
with (root / 'train_original.csv').open() as file:
    rows = list(csv.DictReader(file))
if any(row['qc_status'] == 'needs_review' for row in rows) and not args.allow_review:
    raise SystemExit('QC pending: review penyok01.jpg first. After confirming, update qc_status to confirmed. --allow-review creates provisional outputs with the review flag preserved.')
assert all(row['bottle_id'] in ['01', '02'] and row['split'] == 'train' for row in rows)
assert args.per_class >= 4
rng = random.Random(42)
output = root / 'augmented'
output.mkdir(exist_ok=True)
result = list(rows)
parameters = []
image_cache = {}
for row in rows:
    cached = ImageOps.exif_transpose(Image.open(root / row['filename'])).convert('RGB')
    cached.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    image_cache[row['filename']] = cached
for label in ['normal', 'label_miring', 'label_hilang', 'penyok']:
    sources = [row for row in rows if row['label'] == label]
    for index in range(args.per_class - len(sources)):
        source = sources[index % len(sources)]
        image = image_cache[source['filename']].copy()
        brightness = rng.uniform(0.85, 1.15)
        contrast = rng.uniform(0.90, 1.10)
        blur = rng.uniform(0, 0.35)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = image.filter(ImageFilter.GaussianBlur(blur))
        filename = f"{Path(source['filename']).stem}_aug{index + 1:03d}.jpg"
        destination = output / filename
        image.save(destination, quality=95)
        row = dict(source)
        row.update(filename='augmented/' + filename, is_augmented=1, width=image.width, height=image.height, sha256='', view_id='photometric_augmentation')
        import hashlib
        row['sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
        result.append(row)
        parameters.append(dict(filename=row['filename'], source_filename=source['source_filename'], brightness=brightness, contrast=contrast, blur=blur))
with (root / 'train.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(result)
with (root / 'augmentation_parameters.json').open('w') as file:
    json.dump(parameters, file, indent=2)
assert len(result) == args.per_class * 4
assert len({row['sha256'] for row in result}) == len(result)
print(f'Training samples: {len(result)}, originals: {len(rows)}, augmented: {len(parameters)}')
print('Validation and test were not modified.')
