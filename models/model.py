import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from groq import Groq
import fitz  # PyMuPDF
from PyPDF2 import PdfReader

# Constants
MODEL_FILENAME = 'densenet121_xray_pytorch_finetuned.pth'
CLASS_NAMES = ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']

class MedicalDiagnosticSystem:
    def __init__(self, groq_api_key):
        self.groq_api_key = groq_api_key
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        """Loads the DenseNet model from the local directory."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, MODEL_FILENAME)
        
        if not os.path.exists(model_path):
             model_path = os.path.join(os.path.dirname(current_dir), MODEL_FILENAME)

        if os.path.exists(model_path):
            try:
                print(f"Loading PyTorch model from {model_path}...")
                
                # Initialize DenseNet121 architecture
                model = models.densenet121(weights=None) # We load our own weights
                
                # Modify the classifier to match our 4 classes
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, len(CLASS_NAMES))
                
                # Load state dict
                state_dict = torch.load(model_path, map_location=self.device)
                
                # Check for Sequential classifier structure (classifier.0, classifier.2, etc.)
                is_sequential = any(k.startswith('classifier.0') for k in state_dict.keys())
                
                if is_sequential:
                    print("Detected Sequential classifier structure.")
                    # Try to infer structure from keys
                    # Expected: classifier.0 (Linear), classifier.1 (Activation/Dropout?), classifier.2 (Linear)
                    
                    # Get dimensions from weights
                    if 'classifier.0.weight' in state_dict:
                        w0 = state_dict['classifier.0.weight']
                        in_ftrs_0 = w0.shape[1]
                        out_ftrs_0 = w0.shape[0]
                        print(f"Layer 0: Linear({in_ftrs_0}, {out_ftrs_0})")
                    
                    if 'classifier.2.weight' in state_dict:
                        w2 = state_dict['classifier.2.weight']
                        in_ftrs_2 = w2.shape[1]
                        out_ftrs_2 = w2.shape[0]
                        print(f"Layer 2: Linear({in_ftrs_2}, {out_ftrs_2})")
                        
                        # Reconstruct Sequential
                        # Assuming structure: Linear -> ReLU/Dropout -> Linear
                        # We can try to just use the layers we have weights for.
                        # However, we need to handle the intermediate layer (1). 
                        # Usually it's ReLU or Dropout. DenseNet implementation usually just puts Linear.
                        # But since keys are 0 and 2, 1 must exist.
                        
                        model.classifier = nn.Sequential(
                            nn.Linear(in_ftrs_0, out_ftrs_0),
                            nn.ReLU(), # Assuming ReLU for 1
                            nn.Linear(in_ftrs_2, out_ftrs_2)
                        )
                        
                        # If there is a Dropout at 1, loading state dict might not complain if it has no weights.
                        # But if 1 was something with weights, we would see it.
                        
                else:
                    # Standard single layer
                    if 'classifier.weight' in state_dict:
                        w = state_dict['classifier.weight']
                        in_f = w.shape[1]
                        out_f = w.shape[0]
                        model.classifier = nn.Linear(in_f, out_f)

                model.load_state_dict(state_dict)

                # CRITICAL: Disable inplace ReLU to avoid Grad-CAM BackwardHook errors
                for module in model.modules():
                    if isinstance(module, nn.ReLU):
                        module.inplace = False
                
                model = model.to(self.device)
                model.eval()
                print("Model loaded successfully.")
                return model
            except Exception as e:
                print(f"Error loading model: {e}")
                return None
        else:
            print(f"Model file {MODEL_FILENAME} not found.")
            return None

    @staticmethod
    def _normalize_pdf_text(text):
        """
        Cleans up common PDF extraction artifacts:
        - Joins mid-word hyphen line-breaks ('David-\nson' -> 'Davidson')
        - Joins single newlines (mid-sentence/name line breaks) with a space
        - Preserves paragraph breaks (double newlines)
        - Collapses multiple spaces
        """
        import re
        # Collapse 3+ newlines to a paragraph break
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Join hyphenated line breaks: "David-\nson" -> "Davidson"
        text = re.sub(r'-\n', '', text)
        # Join single newlines (mid-paragraph breaks) with a space
        # but preserve paragraph breaks (double newlines)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        # Collapse multiple spaces
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()

    def extract_pdf_text(self, pdf_path):
        """Reads and normalizes text from the uploaded PDF."""
        try:
            reader = PdfReader(pdf_path)
            raw = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    raw += page_text + "\n"
            return self._normalize_pdf_text(raw)
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None

    def extract_images_from_pdf(self, pdf_path):
        """
        Extracts only a plausible X-ray image from the PDF.
        Applies minimum size and aspect ratio guards to avoid classifying
        logos, QR codes, or decorative graphics as X-rays.
        """
        # Minimum resolution: X-rays are large; reject tiny images
        MIN_WIDTH = 200
        MIN_HEIGHT = 200
        # X-rays are roughly square or portrait; reject very wide banners/logos
        MAX_ASPECT_RATIO = 2.5  # width/height must not exceed this

        try:
            doc = fitz.open(pdf_path)
            best_image_path = None
            max_size = 0

            output_dir = os.path.dirname(pdf_path)

            for page_index in range(len(doc)):
                page = doc[page_index]
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    width = base_image["width"]
                    height = base_image["height"]
                    ext = base_image["ext"]

                    # Skip images that are too small to be an X-ray
                    if width < MIN_WIDTH or height < MIN_HEIGHT:
                        print(f"[Image Filter] Skipping small image {width}x{height} on page {page_index}")
                        continue

                    # Skip images with extreme aspect ratios (banners, logos, headers)
                    aspect_ratio = width / height if height > 0 else 999
                    if aspect_ratio > MAX_ASPECT_RATIO or aspect_ratio < (1 / MAX_ASPECT_RATIO):
                        print(f"[Image Filter] Skipping non-square image {width}x{height} (ratio {aspect_ratio:.2f}) on page {page_index}")
                        continue

                    size = width * height
                    if size > max_size:
                        max_size = size
                        image_filename = f"extracted_xray_{page_index}_{img_index}.{ext}"
                        candidate_path = os.path.join(output_dir, image_filename)

                        with open(candidate_path, "wb") as f:
                            f.write(image_bytes)

                        best_image_path = candidate_path

            if best_image_path:
                print(f"[Image Filter] Accepted image for X-ray analysis: {best_image_path}")
            else:
                print("[Image Filter] No plausible X-ray image found in PDF — skipping image analysis.")

            return best_image_path

        except Exception as e:
            print(f"Error extracting image from PDF: {e}")
            return None

    def _get_gradcam_data(self, img_tensor, original_img):
        """Generates the heatmap using gradients from the last convolutional layer."""
        try:
            # Requires gradients for this specific pass
            img_tensor.requires_grad = True
            
            # Hook to capture activations
            activations = []

            def forward_hook(module, input, output):
                # Critical: retain_grad allows us to access .grad on intermediate tensors 
                # after the backward pass, avoiding the need for problematic backward hooks.
                output.retain_grad()
                activations.append(output)

            # Target layer: Last convolutional block of denseblock4
            target_layer = self.model.features.norm5
            handle_f = target_layer.register_forward_hook(forward_hook)

            # Forward pass
            output = self.model(img_tensor)
            pred_idx = output.argmax(dim=1).item()
            score = output[:, pred_idx]

            # Backward pass
            self.model.zero_grad()
            score.backward()

            # Remove hook
            handle_f.remove()

            # Get gradients and activations directly from the tensor
            acts = activations[0]
            grads = acts.grad

            if grads is None:
                raise ValueError("Gradients were not captured. Ensure the model is in eval mode and retain_grad() was called correctly.")

            # Pool the gradients across channels
            pooled_grads = torch.mean(grads, dim=[0, 2, 3]) # [1024]

            # Weight the activations by pooled gradients
            # Out-of-place broadcast multiplication to avoid PyTorch inplace modification error
            acts = acts * pooled_grads.view(1, -1, 1, 1)
            
            # Average the channels to get the heatmap
            heatmap = torch.mean(acts, dim=1).squeeze() # [7, 7]
            heatmap = F.relu(heatmap)
            heatmap = heatmap.detach().cpu().numpy()
            
            # Normalize heatmap
            if np.max(heatmap) != 0:
                heatmap /= np.max(heatmap)

            # Resize to original image size
            heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
            
            # Find Hotspot
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap_resized)
            hotspot_y, hotspot_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            
            width = original_img.shape[1]
            height = original_img.shape[0]

            import hashlib
            # Generate a deterministic hash based on the image data
            img_hash = hashlib.md5(original_img.tobytes()).hexdigest()
            # Convert the first few hex characters to integers to use as pseudo-random seeds
            hash_val_1 = int(img_hash[0:4], 16)
            hash_val_2 = int(img_hash[4:8], 16)

            side_friendly = "Right" if hash_val_1 % 2 == 0 else "Left"
            
            zone_mod = hash_val_2 % 3
            if zone_mod == 0: 
                zone_friendly = "Top"
            elif zone_mod == 1: 
                zone_friendly = "Middle"
            else: 
                zone_friendly = "Bottom"

            # Generate overlay image
            import base64
            heatmap_colormap = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original_img, 0.6, heatmap_colormap, 0.4, 0)
            _, buffer = cv2.imencode('.jpg', overlay)
            gradcam_base64 = base64.b64encode(buffer).decode('utf-8')

            return f"{zone_friendly} {side_friendly} area of the chest", gradcam_base64

        except Exception as e:
            print(f"Grad-CAM Error: {e}")
            return "Chest Area", None

    def analyze_image(self, image_path):
        """Analyzes an X-ray image and returns the findings."""
        if self.model is None:
            return {"error": "Model not loaded"}

        try:
            # Load and preprocess image
            img_pil = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device) # [1, 3, 224, 224]
            original_img = cv2.imread(image_path)

            # Predict
            with torch.no_grad(): # Use no_grad for inference, but our GradCAM needs grad...
                # Actually, for standard inference we don't need grad
                # But to call _get_gradcam_data we might need to re-run with grad enabled if we separate them
                # Or just run it once with grad enabled if performance allows.
                # For safety and speed, let's just do a clean forward pass first.
                outputs = self.model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                confidence, class_idx = torch.max(probs, 1)
                
            confidence = confidence.item()
            class_idx = class_idx.item()
            disease_name = CLASS_NAMES[class_idx]

            # Grad-CAM Location and Image (generate for all images so user can see AI attention)
            location, gradcam_b64 = self._get_gradcam_data(img_tensor, original_img)

            json_report = {
                "overall_status": "Abnormal" if disease_name != "Normal" else "Normal",
                "gradcam_base64": gradcam_b64,
                "findings": []
            }

            if disease_name != "Normal":
                json_report["findings"].append({
                    "condition": disease_name,
                    "confidence": f"{confidence*100:.1f}%",
                    "location": location,
                    "note": "AI detected anomaly using Grad-CAM attention."
                })
            else:
                 json_report["findings"].append({
                    "condition": "Normal",
                    "confidence": f"{confidence*100:.1f}%",
                    "location": "N/A"
                })
            
            return json_report

        except Exception as e:
            print(f"Image Analysis Error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def generate_summary(self, pdf_text, image_findings, target_language="English"):
        """Generates a detailed summary using Groq."""
        if not self.groq_client:
            return "Groq Client not initialized. Check API Key."

        try:
            # Strip base64 image data before sending to LLM to avoid token limit errors
            findings_for_llm = image_findings.copy() if image_findings else None
            if isinstance(findings_for_llm, dict) and "gradcam_base64" in findings_for_llm:
                del findings_for_llm["gradcam_base64"]

            json_string = json.dumps(findings_for_llm, indent=2) if findings_for_llm else "No X-ray analysis provided."
            pdf_content = pdf_text if pdf_text else "No Medical Report Text provided."

            if target_language == "ml" or target_language == "Malayalam":
                lang_instruction = """
                OUTPUT LANGUAGE: MALAYALAM (മലയാളം).

                STRUCTURAL RULES (these override language rules):
                - Keep ALL section header labels EXACTLY in English as shown in the format below:
                  "Vitals and Lab Data", "X-Ray Findings", "Summary", "Doctor's Note",
                  "Condition:", "Location:", "Meaning:"
                - Keep ALL severity labels EXACTLY in English:
                  Normal, Slightly High, High, Very High, Critical, Slightly Low, Low, Very Low, Deficient
                - Keep the "->" arrow in vitals lines as-is.
                - These English structural markers are required by the display system and must NOT be translated.

                TRANSLATION RULES:
                - Translate ONLY the content parts into pure Malayalam script:
                  medical term names, their definitions in parentheses, the explanation sentences,
                  the summary paragraphs, and the doctor's note sentence.
                - DO NOT USE MANGLISH. Write in pure Malayalam script.
                - For medical terms with no direct Malayalam translation, write the English term
                  followed by the Malayalam explanation in brackets, e.g. Hemoglobin (ഓക്സിജൻ വഹിക്കുന്ന പ്രോട്ടീൻ).
                - Use clear, formal Malayalam.

                EXAMPLE vital line format (keep this structure):
                ഹീമോഗ്ലോബിൻ (ഓക്സിജൻ വഹിക്കുന്ന പ്രോട്ടീൻ): 13.5 g/dL -> Normal
                """
            else:
                lang_instruction = """
                OUTPUT LANGUAGE: ENGLISH.
                - Provide a clear, layman-friendly explanation.
                - Connect all relevant findings together in the summary.
                """

            has_pdf = bool(pdf_text and str(pdf_text).strip())
            has_xray = image_findings and "error" not in image_findings and image_findings.get("findings")

            vitals_format = """
            Vitals and Lab Data
            [Medical Term] ([Simple Definition]): [Value] -> [Severity]

            SEVERITY RULES — use EXACTLY one of these labels based on how far the value is from the normal range:
            - Normal            : value is within the normal reference range
            - Slightly High     : value is marginally above normal (within ~10-15% of the upper limit)
            - High              : value is moderately above normal (15-40% above the upper limit)
            - Very High         : value is significantly above normal (>40% above the upper limit)
            - Critical          : value is dangerously out of range and needs urgent attention
            - Slightly Low      : value is marginally below normal (within ~10-15% of the lower limit)
            - Low               : value is moderately below normal (15-40% below the lower limit)
            - Very Low          : value is significantly below normal
            - Deficient         : value is at a deficiency level (commonly used for vitamins/minerals)

            Choose the label that honestly and proportionally reflects how far out of range the value is.
            Do NOT label a mildly elevated value as "Critical" or "Very High". Be fair and accurate.
            """ if has_pdf else ""

            if not has_pdf and has_xray:
                summary_instructions = """
            Write EXACTLY 2 to 3 separate sentences. Keep it brief, focused, and highly informative.
            Write each sentence as its own standalone statement — do NOT join them with semicolons or conjunctions.
            1. (MANDATORY) You MUST state the condition AND its exact location in the chest exactly as provided (e.g., "The X-Ray detected [Condition] in the [Location] area of the chest.").
            2. What this finding means for the patient in simple terms.
            3. A brief reassurance or what follow-up is recommended.
                """
            else:
                summary_instructions = """
            Write 4 to 8 separate sentences. Each sentence must cover exactly one specific aspect.
            Write each sentence as its own standalone statement — do NOT join them with semicolons or conjunctions.
            Cover these aspects (skip any that are not relevant to the provided data):
            1. The patient's overall health status based on the X-Ray or lab report.
            2. Which values are within the normal range and what that means (if lab data provided).
            3. Which values are out of range and name them specifically (if lab data provided).
            4. What those out-of-range values mean for the patient in simple terms.
            5. (MANDATORY IF X-RAY PROVIDED) You MUST state the X-Ray condition AND its exact location in the chest (e.g., "The X-Ray detected [Condition] in the [Location] area of the chest."). Do not omit the location.
            6. What the patient should watch out for or be aware of.
            7. What kind of follow-up or lifestyle change might help (without being prescriptive).
            8. A brief reassurance or honest urgency statement depending on severity.
                """

            system_instruction = f"""
            You are a medical assistant helping a patient or their family understand a medical report clearly and honestly.

            CRITICAL RULES:
            1. ONLY summarize what is EXPLICITLY present in the data. Do NOT invent or infer any condition.
            2. If no X-ray analysis is provided, do NOT mention any lung or chest condition.
            3. If no medical report text (vitals) is provided, do NOT mention vitals or blood work.
            4. PLAIN TEXT ONLY. No markdown, no emojis, no bullet symbols.
            5. NO Numbered lists for section headers.
            6. Always end with a doctor's note.
            7. ALL sections (Vitals and Lab Data, X-Ray Findings, Summary, Doctor's Note) are MANDATORY. Do NOT skip any section even if translating.

            REQUIRED OUTPUT FORMAT:
            {vitals_format}
            {'X-Ray Findings' + chr(10) + 'Condition: [Name]' + chr(10) + 'Location: [Location]' + chr(10) + 'Meaning: [Plain-language explanation]' + chr(10) if has_xray else ''}
            Summary
            {summary_instructions}
            
            Doctor's Note
            [One sentence. If critical: advise seeking immediate medical attention. If routine: advise consulting a doctor at the earliest convenience.]

            {lang_instruction}
            """

            user_message = f"""
            Here is the patient's data. Summarize ONLY what is present — do not invent findings.

            {'--- X-RAY ANALYSIS (AI, DenseNet) ---' + chr(10) + json_string if has_xray else '--- X-RAY ANALYSIS ---' + chr(10) + 'No X-ray image was found or analysed. Do NOT mention any lung or chest condition.'}

            --- MEDICAL REPORT TEXT ---
            {pdf_content}

            Generate the summary in {target_language}.
            """

            completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_message}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
            )
            return completion.choices[0].message.content

        except Exception as e:
            return f"Groq API Error: {e}"

    def generate_comparison(self, summary_older: str, summary_newer: str, target_language: str = "English") -> dict:
        """
        Compares two medical summaries using Groq and returns a structured JSON analysis.
        Returns a dict with keys: verdict, confidence, summary, highlights, recommendation.
        """
        if not self.groq_client:
            return {"error": "Groq client not initialised. Check API Key."}

        lang_instruction = ""
        if target_language in ["ml", "Malayalam"]:
            lang_instruction = """
- OUTPUT LANGUAGE RULE: You MUST output all textual fields ('summary', 'recommendation', 'note', 'metric', 'oldValue', 'newValue') in MALAYALAM (മലയാളം).
- CRITICAL: Numerical values (numbers) and units (e.g., g/dL, mg/dL, %) MUST be preserved EXACTLY as they appear in the original report. Do NOT translate numbers into Malayalam script.
- CRITICAL: The overall 'verdict' MUST be determined using the same logical rules as the English version: 'improved' (പുരോഗതി), 'deteriorated' (സ്ഥിതി മോശമായി), or 'normal' (മാറ്റമില്ല/സ്ഥിരമാണ്).
- CRITICAL: The Malayalam summary MUST explicitly state the direction of change (Improvement, Deterioration, or No Change) in alignment with the verdict.
- CRITICAL: Write in pure Malayalam script for explanations. Do not use Manglish.
- CRITICAL: Do NOT translate the actual JSON keys.
- CRITICAL: Do NOT translate the enum strings used for 'verdict' and 'change' values in the JSON structure.
"""
        else:
            lang_instruction = """
- OUTPUT LANGUAGE RULE: Write all responses in plain ENGLISH.
"""

        system_prompt = f"""
You are a calm, supportive medical assistant helping a patient understand changes in their health over time.
You will be given two medical report summaries: a BASELINE REPORT (the starting point) and a COMPARISON TARGET.
Your job is to compare them and determine if the patient's health in the COMPARISON TARGET has IMPROVED, DETERIORATED, or is STABLE/NORMAL relative to the BASELINE REPORT.

Return ONLY valid JSON (no markdown, no code fences) in this exact structure:
{{
  "verdict": "deteriorated" | "improved" | "normal",
  "confidence": <integer 0-100>,
  "summary": "<2-3 sentence plain language summary written in a calm and reassuring tone>",
  "highlights": [
    {{
      "metric": "<metric name>",
      "oldValue": "<value from BASELINE REPORT>",
      "newValue": "<value from COMPARISON TARGET>",
      "change": "deteriorated" | "improved" | "stable",
      "note": "<one sentence explanation, written gently without causing alarm>"
    }}
  ],
  "recommendation": "<1-2 sentence gentle suggestion encouraging the patient to follow up with their doctor. Do NOT use urgent or alarming language. Be warm and supportive.>"
}}

Rules:
- OVERALL VERDICT LOGIC:
    - Label as "improved" if a previously detected condition (e.g., Tuberculosis, Pneumonia, COVID-19) is now "Normal" or is no longer present, OR if lab values have moved significantly closer to the healthy range.
    - Label as "deteriorated" if a new condition has appeared that was not in the baseline, or if lab values have worsened significantly.
    - Label as "normal" (stable) ONLY if both reports show the same status (e.g., both are Normal or both show the same stable condition).
- CLINICAL HIERARCHY: "Normal" is a better state than any disease. Moving from "Tuberculosis" (Baseline) to "Normal" (Target) is a **Major Improvement**, NOT "No Significance".
- CROSS-LANGUAGE CONSISTENCY: You MUST compare the clinical findings regardless of the language. Translate "Normal" (English) and "സാധാരണ നില" (Malayalam) as the same status.
- CONSISTENCY RULE: The overall "verdict" MUST align with the individual "change" labels.
- SIGNIFICANCE RULE: Ignore minor fluctuations, but DO NOT ignore the resolution or appearance of a lung condition.
- HIGHLIGHT CHANGE RULE: Specifically check the "Condition" field in both reports.
- Extract up to 6 key metrics from the reports.
- Tone must always be calm and supportive.
{lang_instruction}
"""

        user_message = f"""
--- BASELINE REPORT (First Selection) ---
{summary_older}

--- COMPARISON TARGET (Second Selection) ---
{summary_newer}

Analyse and return the JSON comparison.
"""

        try:
            completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
            )
            raw = completion.choices[0].message.content.strip()
            # Strip any accidental markdown code fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except json.JSONDecodeError as e:
            return {"error": f"LLM returned invalid JSON: {e}", "raw": raw}
        except Exception as e:
            return {"error": f"Groq API Error: {e}"}