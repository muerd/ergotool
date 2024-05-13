import cv2
import mediapipe as mp
import numpy as np
import sys
from math import acos, degrees
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Carga el video
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Contadores e indicadores
# Contadores e indicadores para movimientos y rangos de riesgo
countElbow, countHip = 0, 0
upElbow, downElbow, upHip, downHip = False, False, False, False
ElbowRisk_1, ElbowRisk_2 = 0, 0
HipRisk_1, HipRisk_2, HipRisk_3, HipRisk_4 = 0, 0, 0, 0

# Inicialización de la solución de pose de MediaPipe
with mp_pose.Pose(static_image_mode=False) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)


        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
            #hombro derecho
            x1, y1 = int(landmarks[12].x * width), int(landmarks[12].y * height)
            #hombro izquierdo
            x8, y8 = int(landmarks[11].x * width), int(landmarks[11].y * height)
            #codo derecho
            x2, y2 = int(landmarks[14].x * width), int(landmarks[14].y * height)
            #muñeca derecha
            x3, y3 = int(landmarks[16].x * width), int(landmarks[16].y * height)
            #cadera izquierda
            x4, y4 = int(landmarks[23].x * width), int(landmarks[23].y * height)
            #cadera derecha
            x5, y5 = int(landmarks[24].x * width), int(landmarks[24].y * height)
            #rodilla derecha
            x6, y6 = int(landmarks[26].x * width), int(landmarks[26].y * height)
            #rodilla izquierda
            x7, y7 = int(landmarks[25].x * width), int(landmarks[25].y * height)
            

            

            # Cálculo del ángulo mediante la ley de cosenos para el codo
            p1, p2, p3 = np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3])
            l1, l2, l3 = np.linalg.norm(p2 - p3), np.linalg.norm(p1 - p3), np.linalg.norm(p1 - p2)
            angleElbow = 180 - degrees(acos(max(-1.0, min(1.0, (l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))))            
            # Cálculo del ángulo mediante la ley de cosenos para el la cadera derecha
            p4, p5, p6 = np.array([x1, y1]), np.array([x5, y5]), np.array([x6, y6])
            l4, l5, l6 = np.linalg.norm(p5 - p6), np.linalg.norm(p4 - p6), np.linalg.norm(p4 - p5)
            angleHipR = abs(180 - degrees(acos(max(-1.0, min(1.0, (l4**2 + l6**2 - l5**2) / (2 * l4 * l6))))))
            
            # Cálculo del ángulo mediante la ley de cosenos para el la cadera izquierda
            p7, p8, p9 = np.array([x8, y8]), np.array([x4, y4]), np.array([x7, y7])
            l7, l8, l9 = np.linalg.norm(p8 - p9), np.linalg.norm(p7 - p9), np.linalg.norm(p7 - p8)
            angleHipL = abs(180 - degrees(acos(max(-1.0, min(1.0, (l7**2 + l9**2 - l8**2) / (2 * l7 * l8))))))            
            angleHipMed = (angleHipR + angleHipR) / 2
            
            ####falta contas cambios angulo cadera y todo el resto
            
            
            # Lógica de conteo de movimientos codo
            if angleElbow >= 100:
                upElbow = True                
            if upElbow and not downElbow and angleElbow <= 50:
                downElbow = True
            if upElbow and downElbow and angleElbow >= 100:
                countElbow += 1
                upElbow = downElbow = False
                       
            # Lógica de tiempo en rangos de riesgo del codo
            if angleElbow > 100 or angleElbow < 50:
                ElbowRisk_2 += 1
            else:
                ElbowRisk_1 += 1
            
             # Lógica de conteo de movimientos cadera
            if angleHipMed >= 20:
                upHip = True
            if upHip and not downHip and angleHipMed < 20:
                downHip = True
            if upHip and downHip and angleHipMed >= 20:
                countHip += 1
                upHip = downHip = False
                
            # Lógica de tiempo en rangos de riesgo de la cadera
            if angleHipMed >= 0 and angleHipMed < 20:
                HipRisk_2 += 1
            elif angleHipMed >= 20 and angleHipMed < 60:
                HipRisk_3 += 1
            elif angleHipMed >= 60:
                HipRisk_4 += 1
            else:
                HipRisk_1 += 1
                
                
                
            
            # Dibujar las líneas y los puntos
            #codo
            aux_image = np.zeros(frame.shape, np.uint8)
            cv2.line(aux_image, (x1, y1), (x2, y2), (255, 255, 0), 20)
            cv2.line(aux_image, (x2, y2), (x3, y3), (255, 255, 0), 20)
            cv2.line(aux_image, (x1, y1), (x3, y3), (255, 255, 0), 5)
            contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
            cv2.fillPoly(aux_image, pts=[contours], color=(128, 0, 250))
            
            #cadera
            cv2.line(aux_image, (x5, y5), (x6, y6), (255, 255, 0), 20) #de 24 a 26
            cv2.line(aux_image, (x5, y5), (x1, y1), (255, 255, 0), 20) #de 24 a 12
            cv2.line(aux_image, (x1, y1), (x6, y6), (255, 255, 0), 5)  #de 12 a 26
            contours_1 = np.array([[x5, y5], [x6, y6], [x1, y1]])
            cv2.fillPoly(aux_image, pts=[contours_1], color=(255, 0, 150))

            output = cv2.addWeighted(frame, 1, aux_image, 0.8, 0)
            
            cv2.circle(output, (x1, y1), 6, (0, 255, 255), 4)
            cv2.circle(output, (x2, y2), 6, (128, 0, 250), 4)
            cv2.circle(output, (x3, y3), 6, (255, 191, 0), 4)
            cv2.circle(output, (x5, y5), 6, (200, 205, 155), 4)
            cv2.circle(output, (x6, y6), 6, (255, 0, 191), 4)

            # Mostrar los textos en pantalla
            #mostrar codo
            cv2.putText(output, f"Angle: {int(angleElbow)}", (x2 + 30, y2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 250), 2)
            cv2.putText(output, f"CountCodo: {countElbow}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(output, f"Codo Riesgo 2: {ElbowRisk_2 / 30:.2f} segundos", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(output, f"Codo Riesgo 1: {ElbowRisk_1 / 30:.2f} segundos", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            #mostrar cadera
            cv2.putText(output, f"Angle: {int(angleHipMed)}", (x5 + 30, y5 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 100), 2)
            cv2.putText(output, f"CountCadera: {countHip}", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(output, f"Cadera Riesgo 2: {HipRisk_2 / 30:.2f} segundos", (50,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(output, f"Cadera Riesgo 3: {HipRisk_3 / 30:.2f} segundos", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            cv2.putText(output, f"Cadera Riesgo 4: {HipRisk_4 / 30:.2f} segundos", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(output, f"Cadera Riesgo 1: {HipRisk_1 / 30:.2f} segundos", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("output", output)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar el video y cerrar ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()

# Cálculo de porcentajes de tiempo en riesgo
percentElbowRisk_1 = (ElbowRisk_1 / total_frames) * 100
percentElbowRisk_2 = (ElbowRisk_2 / total_frames) * 100
percentHipRisk_1 = (HipRisk_1 / total_frames) * 100
percentHipRisk_2 = (HipRisk_2 / total_frames) * 100
percentHipRisk_3 = (HipRisk_3 / total_frames) * 100
percentHipRisk_4 = (HipRisk_4 / total_frames) * 100

# Etiquetas y porcentajes para el gráfico
labels = ['Elbow Risk 1', 'Elbow Risk 2', 'Hip Risk 1', 'Hip Risk 2', 'Hip Risk 3', 'Hip Risk 4']
percentages = [percentElbowRisk_1, percentElbowRisk_2, percentHipRisk_1, percentHipRisk_2, percentHipRisk_3, percentHipRisk_4]

# Crear un gráfico de barras para visualizar los porcentajes
plt.figure(figsize=(10, 6))
plt.bar(labels, percentages, color=['green', 'red', 'green', 'cyan', 'orange', 'red'])
plt.xlabel('Risk Categories')
plt.ylabel('Percentage of Total Video Time (%)')
plt.title('Percentage Time Spent in Each Risk Category')
plt.ylim(0, 100)  # Limitar el eje Y para mejorar la visualización
plt.show()
