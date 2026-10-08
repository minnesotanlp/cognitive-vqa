split_prompt = """## Context
You will be given a few continuous screenshots of the video corresponding to approximately 10 seconds of video duration, and provide detailed, faithful, and accurate analysis of this video segment. The objective of this analysis is to group the video into short segments based on the activities for the sake of understanding the intentions behind the actions.

## Instructions
To perform the analysis of decomposing a video into separate activity segments, let’s do it step by step.
1. Based on the provided frames of this video segment, please describe the contents of the video
segment briefly and accurately. You should cover each action and event in the clip. The description should be detailed, faithful, and accurate. It should come with a header: "[1. Description]:".
2. Based on your description, please answer the following question: "Is this video segment a single activity or combination of multiple activities?". The definition of an activity is a segment in which the presenter is working toward a single, nameable unit of progress that a viewer would naturally describe with one verb phrase and seek to independently. It can be an action, for example wiping up a spill. (However, actions that seem *unrelated to the main task*, e.g. taking a sip of a drink, can be ignored.) It can also be an intellectual activity, for example brainstorming solutions or reading the introduction to a paper. It ends when that unit is complete or explicitly abandoned, and a different unit begins. Your answer should come with a header: "[2. Single:]"
3. If the answer to the previous question is "combination of multiple activities", please provide the index of frame(s) separating the scenes from the given frame. Your answer should come with a header: "[3. Frames]:" and in the format of a list of integers. IMPORTANT: frames are numbered starting from 1 (1-indexed). The first frame is frame 1, the second is frame 2, and so on. Use 1-indexed frame numbers in your answer. Do NOT include frame 1 in the list — only report boundaries at frame 2 or later, since frame 1 is always the start of this clip and is handled separately.

## Example
Your response should be in the following format:
[1. Description]: This video shows ...
[2. Single: yes/no]: No.
[3. Frames]: [5, 9]
Please pay special attention to:
- The precise localization of the frames is very important for downstream tasks.
- Frames are 1-indexed: the first frame you are given is frame 1, not frame 0.
- The summarization at the scene level should be consistent with the frames you provided. For instance, the number of scenes should be one more than the number of frames in the list. If you provide 0 frames since the images display a consistent scene, you will give 1 summary; If you provide 1 frame, there should be 2 summaries; if you provide 2 frames, there should be 3 summaries, etc. Now you will be presented the video frames, please perform the analysis carefully.
"""

merge_prompt = """You are going to help with determining if a short video segment is a consistent activity. You will be given a few continuous screenshots of the video clip, along with the accompanying audio, and provide detailed, faithful, and accurate analysis of this video segment. Your objective is simple: if *the video clip starting from the second frame* is a consistent activity with *the first frame*. Answer with "yes" or "no". The definition of an activity is a segment in which the presenter is working toward a single, nameable unit of progress that a viewer would naturally describe with one verb phrase and seek to independently. It can be an action, for example wiping up a spill. (However, actions that seem *unrelated to the main task*, e.g. taking a sip of a drink, can be ignored.) It can also be an intellectual activity, for example brainstorming solutions or reading the introduction to a paper. It ends when that unit is complete or explicitly abandoned, and a different unit begins.  Now you will be presented the video frames, please perform the analysis carefully.
"""