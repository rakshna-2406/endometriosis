print("Testing model import...")
from model import EndoFusionModel
print("Model imported successfully!")

print("Creating model instance...")
model = EndoFusionModel()
print("Model created!")

print("Testing training...")
result = model.train('endometriosis_clinical_data.csv', 'Image')
print("Training result:", result)