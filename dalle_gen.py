from openai import OpenAI
import requests

# client = OpenAI(api_key='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')

# response = client.images.generate(
#   model="dall-e-2",
#   prompt="a white siamese cat",
#   size="1024x1024",
#   quality="standard",
#   n=1,
# )

# image_url = response.data[0].url
# response = requests.get(image_url)

# if response.status_code == 200:
#     filename = "dalle_gen/imagenet/white_siamese_cat_2.jpg"

#     with open(filename, 'wb') as f:
#         f.write(response.content)

#     print(f"Saved to {filename}")
# else:
#     print("Fail...")


client = OpenAI(api_key='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')
saved_path = "dalle_gen/imagenet/white_siamese_cat_2.jpg"

def dalle_gen(client, saved_path, input_text, saved=False):

    response = client.images.generate(
    model="dall-e-2",
    prompt=input_text,
    size="1024x1024",
    quality="standard",
    n=1,
    )

    image_url = response.data[0].url
    response = requests.get(image_url)

    if response.status_code == 200:
        # filename = saved_path
        if saved:
            with open(saved_path, 'wb') as f:
                f.write(response.content)

        print(f"Saved to {saved_path}")
        return saved_path
    else:
        print("Fail...")