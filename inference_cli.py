from neural_kb import predict
from fact_base import get_facts
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to side input image')

args = ap.parse_args()


image_path = args.image

disease_id, label, prediction = predict(image_path)

print(prediction)
print(disease_id)
result = get_facts(int(disease_id))

print(result['disease']['Disease'])
