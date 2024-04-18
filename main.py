# from deepforest import main
# from deepforest import get_data
import matplotlib.pyplot as plt
import deepforest
print(deepforest.__version__)  # This should print the version if it's correctly installed.

# Initialize the model
model = main.deepforest()
model.use_release()

# Load an example image (Replace with your own image path)
image_path = get_data("img.png")

# Predict image
boxes = model.predict_image(path=image_path, return_plot=True)

# Show the image with bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(boxes)
plt.axis('off')
plt.show()
