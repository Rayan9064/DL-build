# Video Labeling Model: Non-Technical Overview

## What This Project Does

This project helps teams label workplace or factory video clips faster.

Instead of asking people to label every clip from scratch, the system watches each video first and suggests an initial label. A human reviewer can then accept, correct, or improve that suggestion in Label Studio.

The goal is not to replace human reviewers. The goal is to reduce repetitive work and make the first pass of video labeling faster.

## The Simple Version

Think of the system as three parts:

1. **Video sampler**
   It takes a small, manageable sample from a very large video dataset.

2. **Labeling workspace**
   Label Studio gives people a browser-based interface where they can watch videos and mark what is happening.

3. **Auto-labeling assistant**
   A pretrained video model looks at a clip and suggests one label, such as `Assemble`, `Inspect`, or `Idle`.

## What Dataset Is Used?

The prototype uses `builddotai/Egocentric-10K` from Hugging Face.

This dataset contains egocentric industrial video clips. “Egocentric” means the video is recorded from a first-person point of view, such as a worker-facing camera or body-mounted camera.

The full dataset is very large, so this project does not download everything. It streams a small sample and saves only a few clips locally for testing and labeling.

## What Labels Does It Start With?

The starter labels are:

- **Assemble**: the person appears to be building, fixing, handling, or putting something together.
- **Inspect**: the person appears to be checking, looking at, reviewing, or observing something.
- **Idle**: the person appears to be waiting, standing, sitting, or doing little visible action.

These are only starter labels. They should be changed to match the real factory process.

Example custom labels might be:

- Pick Part
- Scan Barcode
- Tighten Screw
- Inspect Surface
- Move Cart
- Wait For Machine

## How The Model Makes A Suggestion

For each video clip:

1. The model opens the video.
2. It selects 16 frames spread across the clip.
3. It uses a pretrained action-recognition model to guess what action is happening.
4. It converts that general action guess into one of the team’s labeling categories.
5. It sends the suggested label back to Label Studio.

The suggestion appears as a timeline label that covers the full clip.

## What Model Is Used?

The prototype uses a pretrained video model called `r2plus1d_18`.

This model comes from TorchVision and was pretrained on Kinetics-400, a broad action-recognition dataset.

Because it was trained on general action videos, it will not perfectly understand a specific factory process on day one. It is useful as a starting assistant, and it becomes more valuable after the team collects corrected labels and fine-tunes a model later.

## What A Human Reviewer Does

A reviewer opens Label Studio, watches a video, and sees the model’s suggested label.

The reviewer can:

- Accept the suggestion if it is correct.
- Change it if it is wrong.
- Add better timeline detail if the clip contains multiple actions.

The corrected labels become high-quality training data for future model improvement.

## Why This Is Useful

Manual video labeling can be slow and repetitive. This project speeds up the first pass by giving reviewers a starting point.

Benefits:

- Faster labeling workflow
- Consistent starting labels
- Easy human correction
- Works with local sample clips
- Avoids downloading the entire large dataset
- Creates a path toward a custom factory-specific model

## What It Does Not Do Yet

This prototype does not yet provide a fully trained custom factory model.

Current limitations:

- It predicts one label for the whole clip.
- It uses general pretrained action knowledge.
- It still needs human review.
- It has not yet learned from corrected factory labels.

These are normal limitations for a first prototype.

## How It Improves Over Time

The improvement path is:

1. Start with the pretrained model.
2. Let reviewers correct predictions in Label Studio.
3. Export the corrected labels.
4. Train or fine-tune a model on those corrected examples.
5. Replace the starter mapping with a factory-specific model.

Over time, the system should become more accurate for the exact actions the team cares about.

## What Non-Technical Users Need To Know

Reviewers only need Label Studio.

They do not need to understand Docker, Python, PyTorch, or Hugging Face.

Their job is to:

- Open the assigned video task.
- Watch the clip.
- Review the suggested label.
- Correct it when needed.
- Save the annotation.

## What Technical Users Manage

Technical users handle:

- Starting Docker
- Exporting sample videos
- Connecting Label Studio to the ML backend
- Updating the label list
- Updating the mapping logic
- Exporting corrected labels for future training

## Success Criteria For The Prototype

This prototype is successful if:

- Label Studio opens in the browser.
- Sample videos can be imported.
- The ML backend connects successfully.
- A model prediction appears on a video task.
- Human reviewers can correct and save labels.

## Recommended First Customization

Before using this with a real team, replace the starter labels with labels from the actual factory workflow.

Start by answering:

- What actions do reviewers need to identify?
- Are labels clip-level or timeline-level?
- Can one clip contain multiple actions?
- Which labels are most common?
- Which labels are most important for quality, safety, or operations?

Once those labels are clear, update the Label Studio configuration and model mapping.
