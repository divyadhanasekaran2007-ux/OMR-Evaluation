import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import io
from openpyxl import load_workbook

def get_answer_key(set_choice, excel_file):
    """
    Loads the answer key from the correct sheet of the uploaded Excel file.
    """
    if excel_file is None:
        return None, None
    
    sheet_name = 'Set - A' if set_choice == 'Set A' else 'Set - B'
    
    try:
        key_df = pd.read_excel(io.BytesIO(excel_file.getvalue()), sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Error reading the Excel sheet '{sheet_name}'. Please ensure the sheet name is correct and the file is not corrupted. Error: {e}")
        return None, None

    subjects = key_df.columns.tolist()
    answer_key = {}
    
    for sub in subjects:
        subject_answers = {}
        for item in key_df[sub].dropna():
            try:
                item_str = str(item)
                if ' - ' in item_str:
                    q_str, a_str = item_str.split(' - ')
                else:
                    q_str, a_str = item_str.split('. ')
                
                question_num = int(q_str.strip())
                answer = a_str.strip()
                subject_answers[question_num] = answer
            except (ValueError, IndexError):
                continue
        answer_key[sub] = subject_answers
    
    return answer_key, subjects

def find_filled_answers(image):
    """
    Processes the uploaded OMR sheet image to find filled answers using a structured grid approach.
    """
    try:
        # Convert to grayscale and apply a blur to reduce noise
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply a binary threshold to get a high-contrast image
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for circles based on area and circularity
        circles = []
        for c in contours:
            area = cv2.contourArea(c)
            # Find the bounding box and aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            
            # Check for circular shape and reasonable size
            if area > 50 and 0.8 <= aspect_ratio <= 1.2:
                # Use a circularity check to be more specific
                perimeter = cv2.arcLength(c, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                if 0.6 <= circularity <= 1.4:
                    circles.append(c)
        
        # Sort circles by y then x coordinates to create a logical flow
        circles.sort(key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
        
        user_answers = {}
        if not circles:
            return user_answers
        
        # Assume 100 questions, 4 options each, in 4 columns
        # We need to find the approximate grid locations
        # This part requires robust sorting and grouping
        
        # Group circles into rows
        rows = []
        if circles:
            current_row = [circles[0]]
            (x_prev, y_prev, w_prev, h_prev) = cv2.boundingRect(circles[0])
            for i in range(1, len(circles)):
                (x, y, w, h) = cv2.boundingRect(circles[i])
                # Check if the circle is on the same vertical line
                if abs(y - y_prev) < (h_prev * 0.7):
                    current_row.append(circles[i])
                else:
                    rows.append(current_row)
                    current_row = [circles[i]]
                    y_prev = y
            rows.append(current_row)
            
        # Refine the grouping to ensure each row has a correct number of options (e.g., 4)
        processed_circles = []
        for row in rows:
            row.sort(key=lambda c: cv2.boundingRect(c)[0]) # Sort by x for correct option order
            if len(row) >= 4:
                # Take the first 4 as the most likely answer bubbles
                processed_circles.extend(row[:4])
        
        # Recalculate filled answers
        question_number = 1
        for i in range(0, len(processed_circles), 4):
            question_bubbles = processed_circles[i:i+4]
            if len(question_bubbles) < 4:
                continue

            filled_found = False
            for j, c in enumerate(question_bubbles):
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [c], -1, 255, -1)
                
                # Check pixel density within the circle
                roi = cv2.bitwise_and(thresh, thresh, mask=mask)
                filled_pixel_count = cv2.countNonZero(roi)
                total_pixel_count = cv2.countNonZero(mask)
                
                # Dynamic threshold: a filled bubble has a high ratio of filled pixels
                fill_ratio = filled_pixel_count / total_pixel_count if total_pixel_count > 0 else 0
                
                if fill_ratio > 0.4:
                    if j == 0: user_answers[question_number] = 'a'
                    elif j == 1: user_answers[question_number] = 'b'
                    elif j == 2: user_answers[question_number] = 'c'
                    elif j == 3: user_answers[question_number] = 'd'
                    filled_found = True
                    break
            question_number += 1
            
        return user_answers

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return {}

def score_omr_sheet(user_answers, answer_key, subjects):
    """
    Compares user answers with the answer key and calculates scores.
    """
    subject_scores = {sub: 0 for sub in subjects}
    total_score = 0
    
    start_q = 1
    for subject in subjects:
        subject_key = answer_key.get(subject, {})
        questions_in_subject = len(subject_key)
        end_q = start_q + questions_in_subject
        
        for q_num in range(start_q, end_q):
            correct_answer = subject_key.get(q_num)
            user_answer = user_answers.get(q_num)

            if user_answer and correct_answer and user_answer.strip() == correct_answer.strip():
                subject_scores[subject] += 1
                total_score += 1

        start_q = end_q
    
    return subject_scores, total_score

# --- Streamlit UI ---
st.set_page_config(page_title="OMR Sheet Scorer", layout="wide")
st.title("OMR Sheet Scorer")
st.write("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Upload Answer Key")
    excel_file = st.file_uploader("Upload Excel File with Answer Keys", type=["xlsx", "xls"])
    
    st.header("2. Choose Set")
    set_choice = st.radio("Select the answer key set:", ("Set A", "Set B"))

    st.header("3. Upload OMR Sheet")
    uploaded_omr_sheet = st.file_uploader("Upload OMR sheet image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    st.write("candidate's name")
    student_name = st.text_input("Enter Candidate's Name")

with col2:
    if uploaded_omr_sheet is not None:
        st.header("4. OMR Sheet Preview")
        image = Image.open(uploaded_omr_sheet)
        st.image(image, caption="Uploaded OMR Sheet", use_column_width=True)

        if st.button("Process and Score"):
            if excel_file is None:
                st.error("Please upload the answer key Excel file to continue.")
            else:
                with st.spinner('Processing OMR sheet...'):
                    answer_key, subjects = get_answer_key(set_choice, excel_file)

                    if answer_key and subjects:
                        user_answers = find_filled_answers(image)
                        
                        if user_answers:
                            subject_scores, total_score = score_omr_sheet(user_answers, answer_key, subjects)
                            
                            st.header("Results")
                            st.success("Scoring complete! Here are your results.")
                            
                            results_df = pd.DataFrame(subject_scores.items(), columns=["Subject", "Score"])
                            st.table(results_df)

                            st.metric("Total Score", f"{total_score}/100")
                            f = 'data.xlsx'
                            wb = load_workbook(f)
                            ws = wb.active
                            data = [[student_name, total_score]]
                            for r in data:
                                ws.append(r)    
                            wb.save(f)

                        else:
                            st.warning("Could not find any answers on the OMR sheet. Please check the image quality.")
    else:
        st.info("Please upload an OMR sheet image and the answer key Excel file to get started.")