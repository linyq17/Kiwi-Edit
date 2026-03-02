BACKGROUND_REFERENCE_PROMPT = """
You are a data rater specializing in video background replacement grading. You will be given a **Reference Image**, an **Original Video** (foreground subject), and the **Edited Video** (result). Your task is to evaluate the background replacement effect on a 5-point scale from three perspectives, paying close attention to the preservation of the foreground subject and the fidelity to the reference image.

**Reference Fidelity & Preservation**
1. Background not changed, or the foreground subject is severely damaged/removed.
2. Background changed but bears no resemblance to the reference image; foreground edges are significantly cut off or distorted.
3. Background resembles the reference but lacks key details; foreground is mostly preserved but has noticeable missing parts or artifacts.
4. Background clearly matches the reference image structure and style; foreground subject is fully preserved with only minor edge errors.
5. Perfect execution: The background is an exact semantic and stylistic match to the reference image, and the foreground subject is preserved pixel-perfectly throughout the entire duration.

**Matting Quality & Temporal Stability**
1. Severe flickering; the background or foreground jitters erratically; distinct "boiling" artifacts on edges.
2. Obvious seams, halos, or "green screen" outlines around the subject; background moves unnaturally or freezes while the camera moves.
3. Edges are generally stable but soft/fuzzy; minor flickering in complex areas (e.g., hair, transparent objects); background stability is acceptable.
4. Clean edges with minimal temporal noise; background motion aligns well with camera movement; casual viewers notice no matting errors.
5. Completely seamless composition; hair/transparency details are perfectly matted; background and foreground interact with perfect temporal stability in every frame.

**Visual Harmony & Perspective**
1. Background looks like a flat 2D image pasted behind a 3D subject; severe perspective or lighting mismatch (e.g., shadows point wrong way).
2. Lighting clashes (e.g., sunny background, dark foreground); no depth integration; subject looks "floating."
3. Perspective and scale are roughly correct; lighting is neutral but doesn't explicitly match the new environment’s ambience.
4. Good environmental integration; foreground lighting tones reflect the new background; cast shadows are present and mostly accurate.
5. Photorealistic integration: Depth of field, motion blur, lighting, and color grading of the foreground perfectly match the reference background; the composite looks like a single, raw video capture.

**The second and third score should no higher than first score!!!**

**Example Response Format:**
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Reference Fidelity & Preservation: A number from 1 to 5.
Matting Quality & Temporal Stability: A number from 1 to 5.
Visual Harmony & Perspective: A number from 1 to 5.

**editing instruction is : {prompt}**
**Below are the reference image, original video, and edited video:**
"""

SUBJECT_REFERENCE_PROMPT = """
You are a data rater specializing in reference-guided object manipulation in videos. You will be given a **Reference Image** (the object to insert/swap), an **Original Video**, and the **Edited Video**. Your task is to evaluate the editing effect on a 5-point scale from three perspectives, specifically checking if the new object in the video matches the identity of the reference image.

**Identity Consistency & Compliance**
1. Object not swapped/added, or a completely unrelated object appears.
2. Object is changed, but looks nothing like the reference image (wrong color, shape, or class).
3. Object class is correct, but identity details (texture, specific markings, logos) differ significantly from the reference image.
4. High resemblance to the reference image; correct geometry and texture, with only minor variations in fine details.
5. Perfect identity transfer: The object in the video is indistinguishable from the reference image in terms of texture, structure, and style, while maintaining the correct pose for the scene.

**Temporal Consistency & Texture Fidelity**
1. The new object deforms, melts, or changes shape uncontrollably across frames.
2. Texture "swims" or flickers; resolution drops significantly compared to the rest of the video; object vanishes in some frames.
3. Object is stable in form, but texture details blur or shift slightly during motion; style looks somewhat pasted-on.
4. Object is structurally solid and texture is consistent; minor edge shimmer or noise visible only on close inspection.
5. Completely temporally coherent; the object maintains rigid structure (or appropriate flexibility) and consistent texture details in every single frame, exactly like a real object.

**Physical Integration & Tracking**
1. Object slides around (bad motion tracking); does not follow camera or scene movement; looks like a sticker on the screen.
2. Missing interactions: No shadows, reflections, or occlusion handling (e.g., object appears on top of things that should be in front of it).
3. Motion tracking is decent with slight drift; lighting is flat or generic; occlusion is roughly correct but imprecise.
4. Accurate tracking; lighting and shadows match the scene's direction and intensity; correct occlusion handling.
5. Physically flawless: Motion tracking, perspective changes, motion blur, shadows, reflections, and lighting interactions are indistinguishable from reality; the object feels physically present in the scene.

**The second and third score should no higher than first score!!!**

**Example Response Format:**
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Identity Consistency & Compliance: A number from 1 to 5.
Temporal Consistency & Texture Fidelity: A number from 1 to 5.
Physical Integration & Tracking: A number from 1 to 5.

**editing instruction is : {prompt}**
**Below are the reference image, original video, and edited video:**
"""


import base64
import json, os
import time
import argparse
import base64
from io import BytesIO
import requests
import yaml
from concurrent.futures import ProcessPoolExecutor
import time
from google import genai
import concurrent.futures
from collections import defaultdict

task_score_map = {
    "subject": ["Identity Consistency & Compliance", "Temporal Consistency & Texture Fidelity", "Physical Integration & Tracking"],
    "background": ["Reference Fidelity & Preservation", "Matting Quality & Temporal Stability", "Visual Harmony & Perspective"],
}

def avg_score_by_edited_type(jsonl_path, subtask):
    sum_scores = []
    count = 0
    total_sum = 0
    total_count = 0

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            scores = data.get("scores", [])
            if subtask and data.get("subtask") != subtask:
                continue
            if not scores:
                continue
            # Initialize sum list on first valid row
            if not sum_scores:
                sum_scores = [0] * len(scores)
            for i, score in enumerate(scores):
                sum_scores[i] += score
                total_sum += score
                total_count += 1
            count += 1
    # Average per index
    avg_per_index = [s / count for s in sum_scores]
    # Overall average
    overall_avg = total_sum / total_count if total_count else 0
    print(f"Task: {subtask} File: {jsonl_path}")
    if subtask:
        for i, avg in enumerate(avg_per_index):
            print(f"  {task_score_map[subtask][i]} score: {avg:.4f}")
        print(f"Task: {subtask} average score: {overall_avg:.4f}\n")
    else:
        print(f"Overall average score: {overall_avg:.4f}\n")


def run_gemini_eval(video_path_1, video_path_2, ref_img, sys_prompt, meta_info, idx):
    try:
        client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
        image1 = client.files.upload(file=ref_img)
        video_1 = client.files.upload(file=video_path_1)
        video_2 = client.files.upload(file=video_path_2)
        uploaded_files = [image1, video_1, video_2]
        for i, file_ref in enumerate(uploaded_files):
            while file_ref.state == "PROCESSING":
                time.sleep(2)
                file_ref = client.files.get(name=file_ref.name)
            uploaded_files[i] = file_ref # Update with ACTIVE state
    except Exception as e:
        print(f"Init failed: {e}")
        return None
    for i in range(3):
        try:
            response = client.models.generate_content(
                model=model_id,
                config={
                    "system_instruction": sys_prompt,
                },
                contents=[
                    uploaded_files[0], # Ref Image
                    uploaded_files[1], # First video
                    uploaded_files[2], # Second video
                ]
            )
            result = response.text
            scores = check_format(result)
            if scores:
                print(f"Task {idx} Success on attempt {i+1}")
                return scores, result, meta_info
        except Exception as e:
            print(f"Task {idx} Attempt {i+1} failed: {e}")
            
    return None

def check_format(out):
    try:
        score_list = []
        for line in out.splitlines():
            if line.startswith('Identity Consistency & Compliance:'):
                instruct_score = int(float(line.split(':')[-1].strip()))
                if instruct_score not in range(1, 6):
                    print("score is not in range(1, 6)")
                    return False
                score_list.append(instruct_score)
            elif line.startswith('Temporal Consistency & Texture Fidelity:'):
                vis_score = int(float(line.split(':')[-1].strip()))
                if vis_score not in range(1, 6):
                    print("score is not in range(1, 6)")
                    return False
                score_list.append(vis_score)
            elif line.startswith('Physical Integration & Tracking:'):
                cons_score = int(float(line.split(':')[-1].strip()))
                if cons_score not in range(1, 6):
                    print("score is not in range(1, 6)")
                    return False
                score_list.append(cons_score)
            elif line.startswith('Reference Fidelity & Preservation:'):
                cons_score = int(float(line.split(':')[-1].strip()))
                if cons_score not in range(1, 6):
                    print("score is not in range(1, 6)")
                    return False
                score_list.append(cons_score)
            elif line.startswith('Matting Quality & Temporal Stability:'):
                cons_score = int(float(line.split(':')[-1].strip()))
                if cons_score not in range(1, 6):
                    print("score is not in range(1, 6)")
                    return False
                score_list.append(cons_score)
            elif line.startswith('Visual Harmony & Perspective:'):
                cons_score = int(float(line.split(':')[-1].strip()))
                if cons_score not in range(1, 6):
                    print("score is not in range(1, 6)")
                    return False
                score_list.append(cons_score)
            else:
                pass
        return score_list
    except Exception as e:
        print(e)
        return False

def batch_process_videos(tasks, max_workers=40):
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_gemini_eval, *task) for task in tasks]
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result())
        
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="gemini-3-flash-preview",
                        choices=["gemini-3-flash-preview", "gemini-2.5-pro"],
                        help="Gemini model to use for evaluation")
    parser.add_argument('--video_paths', type=str, nargs='+',
                        default=['./infer_results/xxx/openve'],
                        help="List of video directories to evaluate")
    args = parser.parse_args()
    model_id = args.model_id
    dataset_file = "./benchmark/RefVIE-Bench/refvie_bench.yaml"
    data_root = './benchmark/RefVIE-Bench/'
    for save_dir in args.video_paths:
        exp_name = save_dir.split('/')[-3]
        if os.path.exists(f"{save_dir}/refvie_{model_id}_score.jsonl"):
            continue
        tasks = []
        with open(dataset_file, 'r') as f:
            data = yaml.safe_load(f)
            for idx, item in enumerate(data):
                video_path = data_root + item['src_video']
                prompt = item['instruction']
                ref_img_path = data_root + item['ref_img']
                out_path = str(idx) + "_" + prompt[:10] + "_" + item['src_video'].split('/')[-1]
                out_path = f"{save_dir}/{out_path}"
                subtask = item['edit_type']
                meta_info = {"data": item, "output": out_path, "subtask": subtask}
                if subtask == 'subject':
                    sys_prompt = SUBJECT_REFERENCE_PROMPT.format(prompt=prompt)
                else:
                    sys_prompt = BACKGROUND_REFERENCE_PROMPT.format(prompt=prompt)
                tasks.append((video_path, out_path, ref_img_path, sys_prompt, meta_info, idx))
            res = batch_process_videos(tasks)
            print(res)
            with open(f"{save_dir}/refvie_{model_id}_score.jsonl", 'w') as f:
                for item in res:
                    if item:
                        scores, result, meta_info = item
                        f.write(json.dumps({"scores": scores, "result": result, "meta_info": meta_info, "subtask": meta_info["subtask"]}) + '\n')
    for save_dir in args.video_paths:
        file_path = f"{save_dir}/refvie_{model_id}_score.jsonl"
        for subtask in ['subject', 'background', None]:
            avg_score_by_edited_type(file_path, subtask)
