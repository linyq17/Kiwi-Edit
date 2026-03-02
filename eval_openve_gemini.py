# The prompts are from OpenVE-3M (https://arxiv.org/abs/2512.07826).
GLOBAL_STYLE = """
You are a data rater specializing in grading video style transfer edits. You will be given an input video, a reference style (image or video), and the styled result video. Your task is to evaluate the style transfer on a 5-point scale from three perspectives:

Instruction Compliance
1. Target style absent or clearly wrong.
2. Style shows in a few areas/frames only, or mixed with unrelated styles.
3. Key traits (palette, brushwork, texture) present but patchy or inconsistent across frames.
4. Style reproduced well across almost the whole video; only small local or brief temporal mismatches.
5. Full, faithful transfer: colour, texture, brushwork, and lighting all match the exemplar consistently over the entire duration and space of the video.

Consistency & Detail Fidelity
1. Major objects, layout, or overall motion lost/distorted; original scene barely recognisable.
2. Main subject recognisable, but its size, perspective, motion, or key parts are clearly wrong/missing.
3. Overall structure and motion correct; some local warping, minor omissions, or slight motion jerkiness.
4. Nearly all geometry and motion intact; only slight, non-distracting deformation.
5. All objects, spatial relations, and motion are perfectly kept; only stylistic, harmless distortion.

Visual Quality & Stability
1. Extreme flickering or “boiling” effects; the style is completely unstable frame-to-frame, making the video unwatchable.
2. Significant and distracting flickering or temporal inconsistency in style application.
3. Noticeable but tolerable flicker or texture “boiling”, especially during motion.
4. Largely stable with only minor, subtle flickering visible in areas of complex motion or fine texture.
5. Perfectly stable and temporally coherent; the style appears “stuck” to the scene with no flickering.

Note: The scores for Consistency & Detail Fidelity and Visual Quality & Stability should not be higher than the Instruction Compliance score.

Example Response Format
Brief reasoning: A short explanation of the scores based on the criteria above, no more than 30 words.
Instruction Compliance: A number from 1 to 5.
Consistency & Detail Fidelity: A number from 1 to 5.
Visual Quality & Stability: A number from 1 to 5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


BACKGROUND_CHANGE = """
You are a data rater specializing in grading video background editing. You will be given two videos (before and after editing) and the editing instruction. Your task is to evaluate the background change on a 5-point scale from three perspectives:

Instruction Compliance
1. No change, or background unrelated to prompt, or foreground also replaced/distorted.
2. Background partly replaced or wrong style/content; foreground noticeably altered.
3. Main background replaced but elements missing/extra, or faint spill onto subject edges.
4. Requested background fully present; foreground intact except minute artefacts or small prompt mismatch (e.g. colour tone).
5. Background exactly matches prompt (content, style, placement); all foreground pixels untouched.

Consistency & Detail Fidelity
1. Large tearing, posterisation, or significant temporal artifacts like flickering, jittering edges; edit area obvious at a glance.
2. Clear cut-out halos, colour-resolution gap, or obvious edge instability over time.
3. Blend acceptable but visible on closer look: slight edge blur, or minor temporal instability.
4. Nearly invisible seams; edges are stable across motion, textures aligned, only minor issues when zoomed in.
5. Indistinguishable composite: edges, textures, resolution and colour grading are perfectly continuous and stable throughout the video.

Visual Quality & Stability
1. Severe mismatch: wrong horizon, conflicting light, floating subject, or static background during camera movement.
2. Noticeable inconsistencies in light or scale; incorrect perspective shifts during motion.
3. Overall believable; small errors in shadow, perspective, or minor motion tracking flaws.
4. Lighting, scale, and depth well matched; background tracks convincingly with camera motion.
5. Physically flawless: coherent light, shadows, perspective, and depth throughout.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Consistency & Detail Fidelity: 1-5.
Visual Quality & Stability: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


LOCAL_CHANGE = """
You are a data rater specializing in grading video replacement edits. You will be given two videos (before and after editing) and the editing instructions.

Instruction Compliance
1. Target not replaced or unrelated edit.
2. Partial replacement or wrong class.
3. Largely replaced but with visible remnants or incorrect count/position.
4. Correct replacement with minor attribute errors.
5. Perfect replacement matching class, number, position, scale, pose, motion, and detail.

Consistency & Detail Fidelity
1. Video heavily broken or object flickers uncontrollably.
2. Obvious seams, colour mismatch, or unstable background.
3. Mostly correct but noticeable flicker or lighting inconsistency.
4. Nearly seamless; only tiny temporal artefacts.
5. Completely seamless and temporally stable.

Visual Quality & Stability
1. Severe tracking, lighting, or perspective errors.
2. Missing shadows, poor occlusion, or mismatched motion.
3. Mostly correct with minor inconsistencies.
4. Well-tracked with realistic interactions.
5. Physically flawless integration.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Consistency & Detail Fidelity: 1-5.
Visual Quality & Stability: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


LOCAL_REMOVE = """
You are a data rater specializing in grading video object removal editing.

Instruction Compliance
1. No edit or completely wrong.
2. Wrong object removed or partial removal.
3. Correct object removed with major errors or ghosting.
4. Correct object removed with minor fragments.
5. Perfect removal with everything else untouched.

Visual Quality & Stability
1. Severe artefacts or flickering.
2. Obvious erase marks or jitter.
3. Noticeable temporal inconsistency.
4. Minor edge issues only on close inspection.
5. Perfectly seamless and stable.

Consistency & Detail Fidelity
1. Background badly reconstructed or static.
2. Background shifts or jitters over time.
3. Mostly correct with small flaws.
4. Clean and stable reconstruction.
5. Background perfectly matches original motion and detail.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


LOCAL_ADD = """
You are a data rater specializing in grading video object addition editing.

Instruction Compliance
1. No edit or wrong object added.
2. Partial or wrong addition.
3. Correct object added with major attribute errors.
4. Correct object with minor inaccuracies.
5. Perfect addition with all attributes correct.

Visual Quality & Stability
1. Severe artefacts or flickering.
2. Obvious paste marks or jitter.
3. Noticeable lighting or colour mismatch.
4. Minor edge or temporal artefacts.
5. Perfectly seamless and stable.

Consistency & Detail Fidelity
1. Severe physical errors or occlusion issues.
2. Poor contact, occlusion, or motion.
3. Mostly correct with minor flaws.
4. Realistic shadows, reflections, and motion.
5. Perfect physical and temporal integration.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


SUBTITLES_EDIT = """
You are a data rater specializing in grading instruction-following subtitle edits.

Instruction Compliance
1. Wrong subtitle or no edit.
2. Right action but wrong content or partial edit.
3. Mostly correct with significant errors.
4. Correct with minor inaccuracies.
5. Perfect subtitle edit with zero unintended changes.

Visual Quality & Stability
1. Attributes completely wrong or unreadable.
2. Major deviation from requested attributes.
3. Acceptable but inconsistent placement or style.
4. Minor inaccuracies only.
5. Perfect attribute matching or professional default choice.

Consistency & Detail Fidelity
1. Major video corruption or subtitle damage.
2. Noticeable artifacts or unintended subtitle changes.
3. Minor unintended effects.
4. Almost perfect preservation.
5. Perfect isolation of the edit.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


CAMERA_MULTI_SHOT_EDIT = """
You are a data rater specializing in grading camera shot type alteration edits.

Instruction Compliance
1. Shot type unchanged or wrong.
2. Direction correct but degree wrong.
3. Generally correct but poorly framed.
4. Correct shot with minor framing issues.
5. Perfect shot type and framing.

Visual Quality & Stability
1. Severe distortion or glitches.
2. Distracting jitter or warping.
3. Minor visual flaws.
4. Very stable with tiny artefacts.
5. Perfectly stable and clear.

Consistency & Detail Fidelity
1. Completely different scene.
2. Major illogical changes.
3. Noticeable continuity errors.
4. Highly consistent with minor discrepancies.
5. Perfect consistency and continuity.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


CREATIVE_EDIT = """
You are a data rater specializing in grading instruction-following creative video edits.

Instruction Compliance
1. Instruction ignored.
2. Attempted but fundamentally failed.
3. Generally follows instruction with major deviations.
4. Successful with minor inaccuracies.
5. Perfect creative execution throughout.

Visual Quality & Stability
1. Unwatchable due to flicker or artefacts.
2. Obvious temporal inconsistency or seams.
3. Mostly stable with noticeable boiling.
4. Very stable with subtle artefacts.
5. Perfectly seamless and stable.

Consistency & Detail Fidelity
1. Severe physical inconsistencies.
2. Major lighting or motion errors.
3. Mostly believable with minor flaws.
4. Realistic interaction and preserved details.
5. Indistinguishable from real footage.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


import base64
import json, os
import time
import argparse
import base64
from io import BytesIO
import csv
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import time
from google import genai
import json
from collections import defaultdict

def avg_score_by_edited_type(jsonl_path):
    score_sum = defaultdict(int)
    score_count = defaultdict(int)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                edited_type = record.get("edited_type")
                scores = record.get("scores", [])
                if edited_type and scores:
                    score_sum[edited_type] += sum(scores)
                    score_count[edited_type] += len(scores)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_num}: {e}")
    # compute averages
    avg_scores = {
        edited_type: score_sum[edited_type] / score_count[edited_type]
        for edited_type in score_sum
        if score_count[edited_type] > 0
    }
    print(score_count)
    return avg_scores

prompt_type = {
    'global_style': GLOBAL_STYLE,
    'local_change': LOCAL_CHANGE,
    'background_change': BACKGROUND_CHANGE,
    'local_remove': LOCAL_REMOVE,
    'local_add': LOCAL_ADD,
}

def run_two_videos_gemini(video_path_1, video_path_2, sys_prompt, meta_info, idx):
    try:
        client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

        # 1. Upload both videos
        # print("Uploading videos...")
        video_1 = client.files.upload(file=video_path_1)
        video_2 = client.files.upload(file=video_path_2)
        
        # 2. Wait for both to be ACTIVE
        uploaded_files = [video_1, video_2]
        for i, file_ref in enumerate(uploaded_files):
            while file_ref.state == "PROCESSING":
                time.sleep(2)
                file_ref = client.files.get(name=file_ref.name)
            uploaded_files[i] = file_ref # Update with ACTIVE state
    except Exception as e:
        print(f"Init failed: {e}")
        return None
    # 3. Retry loop (3 attempts)
    for i in range(3):
        try:
            response = client.models.generate_content(
                model=model_id,
                config={
                    "system_instruction": sys_prompt,
                    # "max_output_tokens": 1024,
                },
                contents=[
                    uploaded_files[0], # First video
                    uploaded_files[1], # Second video
                    # "Compare these two videos and provide the requested scores."
                ]
            )
            
            result = response.text
            scores = check_format(result)
            
            if scores:
                print(f"Task {idx} Success on attempt {i+1}")
                return scores, result, meta_info['prompt'], meta_info['edited_type']
                
        except Exception as e:
            print(f"Task {idx} Attempt {i+1} failed: {e}")
            
    return None

def check_format(out):
    try:
        for line in out.splitlines():
            if line.startswith('Instruction Compliance:'):
                instruct_score = int(float(line.split(':')[-1].strip()))
                if instruct_score not in range(1, 6):
                    print("score is not in range(1, 6)")
                    return False
            elif line.startswith('Visual Quality & Stability:'):
                vis_score = int(float(line.split(':')[-1].strip()))
                if vis_score not in range(1, 6):
                    print("score is not in range(1, 6)")
                    return False
            elif line.startswith('Consistency & Detail Fidelity:'):
                cons_score = int(float(line.split(':')[-1].strip()))
                if cons_score not in range(1, 6):
                    print("score is not in range(1, 6)")
                    return False
            else:
                pass
        return [instruct_score, vis_score, cons_score]
    except Exception as e:
        print(e)
        return False

def batch_process_videos(tasks, max_workers=20):
    """
    tasks: List of tuples [(v1, v2, prompt, key), ...]
    """
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_two_videos_gemini, *task) for task in tasks]
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result())
        
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="gemini-2.5-pro",
                        choices=["gemini-3-flash-preview", "gemini-2.5-pro"],
                        help="Gemini model to use for evaluation")
    parser.add_argument('--video_paths', type=str, nargs='+',
                        default=['./infer_results/xxx/openve'],
                        help="List of video directories to evaluate")
    args = parser.parse_args()
    model_id = args.model_id
    video_paths = args.video_paths
    for save_dir in video_paths:
        if os.path.exists(f"{save_dir}/openve_{model_id}_score.jsonl"):
            continue
        tasks = []
        with open('./benchmark/OpenVE-Bench/benchmark_videos.csv', 'r') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                prompt = row['prompt']
                edited_type = row['edited_type']
                if row['edited_type'] in ['creative_edit', 'camera_edit', 'subtitle_edit']:
                    continue
                out_path = row['original_video'].split('/')[-1].replace('.mp4', '_edited.mp4')
                edited_video_path =  f"{save_dir}/{out_path}"
                video_path = './benchmark/' + row['original_video']
                sys_prompt = prompt_type[edited_type].format(edit_prompt=prompt)
                meta_info = {
                    "prompt": prompt, "edited_type": edited_type
                }
                tasks.append((video_path, edited_video_path, sys_prompt, meta_info, idx))
                
            res = batch_process_videos(tasks)
            print(res)
            with open(f"{save_dir}/openve_{model_id}_score.jsonl", 'w') as f:
                for item in res:
                    if item:
                        scores, result, prompt, edited_type = item
                        f.write(json.dumps({"scores": scores, "prompt": prompt, "edited_type": edited_type, "result": result}) + '\n')

    for save_dir in video_paths:
        file_path = f"{save_dir}/openve_{model_id}_score.jsonl"
        averages = avg_score_by_edited_type(file_path)
        all_scores = sum(averages.values()) / len(averages)
        print(f"{file_path} All scores: {all_scores:.2f}")
        for edited_type, avg in averages.items():
            print(f"  {edited_type}: {avg:.2f}")
