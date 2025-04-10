from imports import *

class mediaPipeDeclaration:

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    @staticmethod
    def get_pose_landmarker():
        options = mediaPipeDeclaration.PoseLandmarkerOptions(
                  base_options=mediaPipeDeclaration.BaseOptions(model_asset_path='Pose_Landmarks/pose_landmarker_full.task'),
                  running_mode=mediaPipeDeclaration.VisionRunningMode.VIDEO
                )
        return mediaPipeDeclaration.PoseLandmarker.create_from_options(options)

    @staticmethod
    def draw_landmarks_on_image(rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        for pose_landmarks in pose_landmarks_list:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        return annotated_image