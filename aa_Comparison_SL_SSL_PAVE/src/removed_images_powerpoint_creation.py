from pptx import Presentation
from pptx.util import Inches, Pt
from pathlib import Path
from pptx.enum.text import PP_ALIGN

# Directory containing the images
base_directory = Path("/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms")

# Subdirectories for different image sets
directories = ['CC_ST_Relationships', 'HPV_Relationships']

# Create a presentation object
presentation = Presentation()

# Loop through each directory and add images to the presentation
for dir_name in directories:
    image_directory = base_directory / dir_name
    for image_path in image_directory.glob("*.png"):
        # Get image filename without extension for the slide title
        title = image_path.stem

        # Create a slide with title and image
        slide_layout = presentation.slide_layouts[5]  # Layout 5 is a blank slide
        slide = presentation.slides.add_slide(slide_layout)

        # Set the title of the slide
        title_box = slide.shapes.title
        title_box.text = title

        # Adjust the font size and alignment of the title
        title_box.text_frame.paragraphs[0].font.size = Pt(12)
        title_box.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER  # Center align the title

        # Add the image to the slide
        slide.shapes.add_picture(str(image_path), Inches(1), Inches(1.5), width=Inches(8.5), height=Inches(6))

# Save the presentation
output_path = "/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/removed_images_test2_brian_analysis_presentation.pptx"
presentation.save(output_path)
