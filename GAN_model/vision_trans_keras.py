from vit_keras import vit, utils

image_size = 384
classes = utils.get_imagenet_classes()
model = vit.vit_b16(
    image_size=image_size,
    activation='sigmoid',
    pretrained=True,
    include_top=True,
    pretrained_top=True
)
url = 'https://upload.wikimedia.org/wikipedia/commons/d/d7/Granny_smith_and_cross_section.jpg'
image = utils.read(url, image_size)
X = vit.preprocess_inputs(image).reshape(1, image_size, image_size, 3)
y = model.predict(X)
print(classes[y[0].argmax()]) # Granny smith
