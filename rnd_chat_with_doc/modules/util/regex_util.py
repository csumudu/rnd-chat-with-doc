import re
from typing import List


def extract_image_paths(text: str) -> List[str]:
    # Regular expression pattern to match image paths in the markdown format
    pattern = r"!\[.*?\]\((.*?)\)"

    # Find all matches in the text
    image_paths = re.findall(pattern, text,re.DOTALL)

    return image_paths


if __name__ == "__main__":
    text = """
            This is an example text with images:
            ![config guide cover page](../GeneratedDocs/images/config_guide_cover_page.png)
            Here is another image:
            ![another image](../GeneratedDocs/images/another_image.png)
            """

    # Extract image paths
    image_paths = extract_image_paths(text)

    # Output the result
    print(image_paths)
