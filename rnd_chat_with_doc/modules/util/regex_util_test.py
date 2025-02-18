from rnd_chat_with_doc.modules.util.regex_util import extract_image_paths


def test_extract_image_paths():
    text = """
            This is an example text with images:
            ![config guide cover page](../GeneratedDocs/images/config_guide_cover_page.png)
            Here is another image:
            ![another image](../GeneratedDocs/images/another_image.png)
             * 8\. Product Configuration
                * 8.1. Operations for Purchased Products
                * 8.2. Configuration of Multiple Product Offers with Same Display Name
                * 9\. Configuring Order Task Framework
                * 9.1. Configuration for executing Tasks in Customer Connect
                * 9.2. Different Order Stages
                * 9.3. Meaning of Special Value Types
                * 10\. References

                ![config guide cover
                page](../GeneratedDocs/images/settings.png)

                # Copyright Notice

                **Release Date**

                Thursday, 1 August 2024
            """

    image_paths = extract_image_paths(text=text)
    assert len(image_paths) == 3
    assert image_paths[0] == "../GeneratedDocs/images/config_guide_cover_page.png"
    assert image_paths[2] == "../GeneratedDocs/images/settings.png"
