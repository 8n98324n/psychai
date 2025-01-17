import cv2
import numpy as np
import mediapipe as mp
import plotly.graph_objects as go
import plotly.express as px


"""

"""

def get_face_with_feature_points(file_of_face, landmarks_list):
        #
        PRESENCE_THRESHOLD = 0.5
        VISIBILITY_THRESHOLD = 0.5

        imag = cv2.imread(file_of_face, cv2.COLOR_RGB2BGR)
        imag = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5) as face_mesh:
            image = cv2.imread(file_of_face)
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            width = image.shape[1]
            height = image.shape[0]
            face_landmarks = results.multi_face_landmarks[0]
            ldmks = np.zeros((468, 3), dtype=np.float32)
            for idx, landmark in enumerate(face_landmarks.landmark):
                if ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                        or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                    ldmks[idx, 0] = -1.0
                    ldmks[idx, 1] = -1.0
                    ldmks[idx, 2] = -1.0
                else:
                    coords = mp_drawing._normalized_to_pixel_coordinates(
                        landmark.x, landmark.y, width, height)
                    if coords:
                        ldmks[idx, 0] = coords[0]
                        ldmks[idx, 1] = coords[1]
                        ldmks[idx, 2] = idx
                    else:
                        ldmks[idx, 0] = -1.0
                        ldmks[idx, 1] = -1.0
                        ldmks[idx, 2] = -1.0

        filtered_ldmks = []
        if landmarks_list is not None:
            for idx in landmarks_list:
                filtered_ldmks.append(ldmks[idx])
            filtered_ldmks = np.array(filtered_ldmks, dtype=np.float32)
        else:
            filtered_ldmks = ldmks    
        image = cv2.imread(file_of_face, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig = px.imshow(image)
        for l in filtered_ldmks:
            name = 'feature_' + str(int(l[2]))
            fig.add_trace(go.Scatter(x=(l[0],), y=(l[1],), name=name, mode='markers', 
                                    marker=dict(color='red', size=5)))
        fig.update_xaxes(range=[0,image.shape[1]])
        fig.update_yaxes(range=[image.shape[0],0])
        fig.update_layout(paper_bgcolor='#eee') 
        return fig

    
    


