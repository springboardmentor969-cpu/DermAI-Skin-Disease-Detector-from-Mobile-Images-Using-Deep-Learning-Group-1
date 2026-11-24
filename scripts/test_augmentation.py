from preprocessing.augmentation import (
    get_train_transforms,
    get_validation_transforms,
    get_test_transforms
)

print("=" * 60)
print("DERMAI DATA AUGMENTATION")
print("=" * 60)

print("\nTraining augmentation:")
print(get_train_transforms())

print("\nValidation transformation:")
print(get_validation_transforms())

print("\nTest transformation:")
print(get_test_transforms())

print("\n" + "=" * 60)
print("DATA AUGMENTATION CONFIGURATION COMPLETE")
print("=" * 60)