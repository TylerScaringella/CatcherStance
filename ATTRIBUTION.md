# AI Tool Usage

## Codex
Codex was used to generate the majority of the web interface and backend API code.

**What it generated**:
- Full-stack web interface for managing the AI pipeline
- Backend API implementation
- README's and text to better describe the purpose and structure of individual module's and folders within the project

**Iterative Development**:
Not everything was created in a single prompt, rather:
- Features were added incrementally through follow-up prompts
- The interface changed based on personal wants and desires during testing
- The backend was re-prompted to migrate from `http` to `Flask`
- Mobile friendly / responsive design


## ChatGPT
ChatGPT was used as my personal consultant throughout the entire project for:
- System design
- ML pipeline architecture
- Debugging and implementation
- Data processing and modeling guidance

**1. Downloader**
- Designed workflow:
    - Extracting TruMedia video URLs
    - Handling authentication/session constraints
    - Scaling download throughput

**2. Detection Pipeline**
*With the YOLO Pose detection model, I needed to extract just the catcher's keypoints from the result. It took many iterations of tweaking different catcher detection configuration values, much of which was done by debugging with Codex and ChatGPT based on the outputs from the model to determine why the model was rejecting the catcher or identifying the wrong person.*
- Advised on:
    - Catcher identification
    - Handling missing frames
- Guided model design:
    - MLP vs. sequence models
    - Dataset splitting and preprocessing

**3. Curator**
- Dataset construction guidance:
    - Labeling entire videos vs. individual frames
    - Structuring fixed-length input sequences
- Feature normalization