
import pygame
from cvzone.PoseModule import PoseDetector
import cv2
from ultralytics import YOLO
import torch
import math
import cvzone
from collections import deque
import imutils
import numpy as np
import time



######################     Required Functions      #####################################
# %%writefile drill_utils.py

def get_target_points(mode,ball_w,body_lm):
    if mode == "Left Hand Dribble - Warm Up":
        target1 = (body_lm[24][0] - (ball_w), body_lm[24][1])
        target2 = ("ground", body_lm[28][1])

        return [(target1,target2)], "fixed"
        
    elif mode == "Left Hand Dribble - Mid":
        target1 = (body_lm[24][0] - int(ball_w*1.5), (body_lm[26][1] + body_lm[24][1])//2)
        target2 = ("ground", body_lm[28][1])
        return [(target1,target2)], "fixed"

    elif mode == "Left Hand Dribble - Low":
        target1 = (body_lm[26][0] - int(ball_w*1.5), body_lm[26][1])
        target2 = ("ground", body_lm[28][1])
        return [(target1,target2)], "fixed"

    if mode == "Right Hand Dribble - Warm Up":
        target1 = (body_lm[23][0] + (ball_w), body_lm[24][1])
        target2 = ("ground", body_lm[28][1])

        return [(target1,target2)], "fixed"
        
    elif mode == "Right Hand Dribble - Mid":
        target1 = (body_lm[23][0] + int(ball_w), (body_lm[26][1] + body_lm[24][1])//2)
        target2 = ("ground", body_lm[28][1])
        return [(target1,target2)], "fixed"

    elif mode == "Right Hand Dribble - Low":
        target1 = (body_lm[25][0] + int(ball_w), body_lm[26][1])
        target2 = ("ground", body_lm[28][1])
        return [(target1,target2)], "fixed"

    elif mode == "Cross Over Drill":
        target1 = (body_lm[24][0] - int(ball_w*1.2),  (body_lm[26][1] + body_lm[24][1])//2)
        target2 = ((body_lm[24][0] + body_lm[23][0])//2 , body_lm[28][1])
        target3 = (body_lm[23][0] + int(ball_w*1.2),  (body_lm[26][1] + body_lm[24][1])//2)
        return [(target1,target2,target3)], "fixed"

    

class GetTargetInOrder:
    def __init__(self,targets,target_type):
        self.target_type = "fixed"
        self.targets = targets[0] if self.target_type == "fixed" else targets
        self.final_target_idx = 0
        self.start_time = 0
        self.speed = 0
        self.n_reps = 0

    def get_target_ord(self, ball_in_target=False):
        if self.target_type == "fixed":
            if ball_in_target:
                if self.n_reps == 0 and self.final_target_idx == 0:
                    self.start_time = time.time()
                if self.final_target_idx + 1 < len(self.targets):
                    self.final_target_idx += 1
                else:
                    self.final_target_idx = 0
                    self.speed = 1/(time.time() - self.start_time)
                    self.n_reps += 1
                    self.start_time = time.time()
    
            current_target_area = self.targets[self.final_target_idx]
            other_targets =self.targets[self.final_target_idx+1:] + self.targets[:self.final_target_idx] if self.final_target_idx+1 < len(self.targets) else self.targets[:self.final_target_idx] 

            return current_target_area, other_targets,self.speed,self.n_reps


def get_distance(target_pos, ball_pos):
    x1,y1 = target_pos
    x2,y2 = ball_pos
    if target_pos[0] == "ground":
        return abs(y2-y1)
        
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance
            

def detectObject(imgMain):
    """Function to detect objects in the input image using YOLO."""
    results = model(imgMain, stream=False, verbose=False,device=device)
    objects = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding Box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100  # Confidence
            cls = int(box.cls[0])  # Class Name
            
            if conf > confidence and (cls == 0 or cls == 2):
                area = (x2 - x1) * (y2 - y1)
                center = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
                # if area < 5000:
                # Draw the detected object's class name, confidence, and bounding box on the image
                cvzone.putTextRect(imgMain, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=[0, 245, 0],
                                   colorT=(255, 255, 255), colorR=[0, 245, 0], offset=5)
                cvzone.putTextRect(imgMain, f'{area}',
                                   (max(0, x1), max(35, y2)), scale=1, thickness=1, colorB=[0, 245, 0],
                                   colorT=(255, 255, 255, 255), colorR=[0, 245, 0], offset=5)
                cv2.rectangle(imgMain, (x1, y1), (x2, y2), (0, 245, 0), 3)
                objects.append([x1, y1, x2, y2, area, center, imgMain])

        return imgMain, objects



###########################################################################################################################


yoloModelPath = "bascket_ball_N.pt"
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
classNames = ['Basket Ball',"rim","Sports Ball"]
confidence = 0.65
model = YOLO(yoloModelPath)


# Initialize the PoseDetector class with the given parameters
detector = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=False,
                        smoothSegmentation=True,
                        detectionCon=0.75,
                        trackCon=0.65)

drills = ["Left Hand Dribble - Warm Up",
          "Left Hand Dribble - Mid",
          "Left Hand Dribble - Low",
          "Right Hand Dribble - Warm Up",
          "Right Hand Dribble - Mid",
          "Right Hand Dribble - Low",
          "Cross Over Drill"]


class BasketballGameMenu:
    def __init__(self,drills,camera_idx=0):
        # Initialize pygame
        pygame.init()

        # Constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 1920, 1080
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 128, 0)
        self.HOVER_GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 128)
        self.HOVER_BLUE = (0,0, 255)
        self.FONT_SIZE = 36
        self.DRILL_BUTTON_WIDTH = self.SCREEN_WIDTH // 2 - 70
        self.BUTTON_HEIGHT = self.SCREEN_HEIGHT // 10
        self.BUTTON_SPACING_X = 20
        self.BUTTON_SPACING_Y = 20
        self.MAX_BUTTON_ROWS = 3
        self.TOTAL_DRILLS = len(drills)
        self.CAMERA_IDX = camera_idx
        # Initialize the display
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Basketball Game Menu")
        self.background_image = pygame.image.load("bg_basketball.png")  # Replace "background.jpg" with your image file
        
        # Create a font for the menu
        self.font = pygame.font.Font(None, self.FONT_SIZE)

        # Create a clock to control timing
        self.clock = pygame.time.Clock()

        # Initialize the camera
        # self.cap = cv2.VideoCapture(1)  # Use the appropriate camera index (0 for the default camera)

        # Timing variables
        self.show_image = False
        
        # Track the currently hovered button
        self.hovered_button = None
        self.selected_drill = None
        self.current_page = 0  # Track the current page of drills

        # List of drills
        self.drill_list = drills.copy()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.hovered_button and self.hovered_button[1] in self.drill_list:
                    self.show_image = True
                    self.start_time = pygame.time.get_ticks()
                    self.selected_drill = self.hovered_button[1]
                elif self.hovered_button and self.hovered_button[1] == "Next":
                    if self.current_page < len(self.drill_list) // (self.MAX_BUTTON_ROWS * 2):
                        self.current_page += 1
                        self.hovered_button = None  # Reset hovered_button when changing the page
                elif self.hovered_button and self.hovered_button[1] == "Previous":
                    if self.current_page > 0:
                        self.current_page -= 1
                        self.hovered_button = None  # Reset hovered_button when changing the page

    def draw_buttons(self):
        # Calculate the range of drills to display on the current page
        start_index = self.current_page * (self.MAX_BUTTON_ROWS * 2)
        end_index = min(start_index + (self.MAX_BUTTON_ROWS * 2), len(self.drill_list))

        for i in range(start_index, end_index):
            row = (i - start_index) // 2
            col = (i - start_index) % 2
            button_x = col * (self.DRILL_BUTTON_WIDTH + self.BUTTON_SPACING_X)
            button_y = self.SCREEN_HEIGHT // 2 - (self.BUTTON_HEIGHT + self.BUTTON_SPACING_Y) + row * (self.BUTTON_HEIGHT + self.BUTTON_SPACING_Y)
            button = pygame.Rect(button_x, button_y, self.DRILL_BUTTON_WIDTH, self.BUTTON_HEIGHT)

            if button.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(self.screen, self.HOVER_BLUE, button)
                self.hovered_button = (button, self.drill_list[i])
            else:
                pygame.draw.rect(self.screen, self.BLUE, button)

            text = self.font.render(self.drill_list[i], True, (255, 255, 255))
            text_rect = text.get_rect(center=button.center)
            self.screen.blit(text, text_rect)

        # Display "Previous" button at the bottom
        previous_button = pygame.Rect(
            (self.SCREEN_WIDTH - self.DRILL_BUTTON_WIDTH) // 2,
            self.SCREEN_HEIGHT - self.BUTTON_HEIGHT - self.BUTTON_SPACING_Y,
            self.DRILL_BUTTON_WIDTH, self.BUTTON_HEIGHT)

        if previous_button.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(self.screen, self.HOVER_GREEN, previous_button)
            self.hovered_button = (previous_button, "Previous")
        else:
            pygame.draw.rect(self.screen, self.GREEN, previous_button)

        previous_text = self.font.render("Previous", True, (255, 255, 255))
        previous_text_rect = previous_text.get_rect(center=previous_button.center)
        self.screen.blit(previous_text, previous_text_rect)

        # Display "Next" button at the bottom
        next_button = pygame.Rect(
            (self.SCREEN_WIDTH - self.DRILL_BUTTON_WIDTH) // 2,
            self.SCREEN_HEIGHT - self.BUTTON_HEIGHT * 2 - self.BUTTON_SPACING_Y * 2,
            self.DRILL_BUTTON_WIDTH, self.BUTTON_HEIGHT)

        if next_button.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(self.screen, self.HOVER_GREEN, next_button)
            self.hovered_button = (next_button, "Next")
        else:
            pygame.draw.rect(self.screen, self.GREEN, next_button)

        next_text = self.font.render("Next", True, (255, 255, 255))
        next_text_rect = next_text.get_rect(center=next_button.center)
        self.screen.blit(next_text, next_text_rect)

    def display_live_camera_feed(self, selected_drill):
        # Display the live camera feed and selected drill
        if selected_drill:
            cap = cv2.VideoCapture(self.CAMERA_IDX)
            # cap = cv2.VideoCapture("hassan-cam.mp4")
            # cap.set(cv2.CAP_PROP_POS_FRAMES,100)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            points = []
            start_time = time.time()
            counter = 0
            drill_mode = selected_drill

            ball_marker_img = cv2.imread("ball_marker2.png", cv2.IMREAD_UNCHANGED)
            target_frame = cv2.imread("target.png", cv2.IMREAD_UNCHANGED)
            target_frame_b = cv2.imread("ball_path.png", cv2.IMREAD_UNCHANGED)
            player_shadow = cv2.imread("player_shadow.png", cv2.IMREAD_UNCHANGED)
            player_shadow = imutils.resize(player_shadow,height=int(frame_height*0.7))
            shadow_h,shadow_w,_ = player_shadow.shape
            
            
            opacity = 150
            target_mask = target_frame_b[:, :, 3] == 255 
            target_frame_b[target_mask, 3] = opacity

            shadow_mask = player_shadow[:, :, 3] == 255 
            player_shadow[shadow_mask, 3] = opacity
            
            start = False
            while True:
                success, img = cap.read()
                c_time = time.time()
                img = cv2.flip(img,1)
                org_img = img.copy()

                if not start:
                    img = detector.findPose(org_img,draw=False)
                    lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)
                    if len(lmList) > 0:
                        start = True

                    else:
                        img = cvzone.overlayPNG(img,player_shadow,pos=((frame_width-shadow_w)//2,(frame_height-shadow_h)//2))

                        
                else:
                    if c_time - start_time > 1 and counter < 5:
                        start_time = time.time()
                        counter += 1
            
                    if counter < 5:
                        cv2.circle(img,(frame_width//2,(frame_height//2)-30),50,(0,0,0),-1)
                        cv2.putText(img,f"{counter}",((frame_width//2)-30,(frame_height//2)),2,3,(0,0,255),3)
        
                    else:
                        imgWithObject, objects = detectObject(img)
                        img = detector.findPose(org_img,draw=False)
                        lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)
    
                        if not len(lmList) > 0:
                            self.show_image = False
                            self.selected_drill = None
                            cap.release()
                            break
                        
                        if len(points) == 0 and len(objects) > 0:
                            ball_w = objects[0][2] - objects[0][0]
                            points, target_type = get_target_points(drill_mode,ball_w,lmList)
                            getTargetInOrd = GetTargetInOrder(points,target_type)
                            current_point, other_points,speed,n_reps = getTargetInOrd.get_target_ord(False)
                            ball_marker_img = cv2.resize(ball_marker_img,(ball_w+int(ball_w*0.4),ball_w+int(ball_w*0.4)))
                            target_frame = cv2.resize(target_frame,(int(ball_w*1.5),int(ball_w*1.5)))
                            target_frame_b= cv2.resize(target_frame_b,(int(ball_w*1.5),int(ball_w*1.5)))
                            target_frame_shape = target_frame.shape
                        
                        elif len(objects) > 0:
                            ball_distance = get_distance(current_point,objects[0][5])
                            current_point, other_points,speed,n_reps = getTargetInOrd.get_target_ord(ball_distance < ball_w)
                            img = cv2.putText(img, f"Speed: {round(speed,2)}", (100, 200), 1,5,(0, 255, 0), 2)
                            img = cv2.putText(img, f"N-Reps: {n_reps}", (100, 300), 1,5,(0, 255, 0),2)
    
                            if not current_point[0] == "ground":
                                img = cvzone.overlayPNG(img,target_frame,pos=(current_point[0]-target_frame_shape[0]//2,current_point[1]-target_frame_shape[0]//2))
                            
                            for target in other_points:
                                if not target[0] == "ground":
                                    img = cvzone.overlayPNG(img,target_frame_b,pos=(target[0]-target_frame_shape[0]//2,target[1]-target_frame_shape[0]//2))
                                
                            
                            img = cvzone.overlayPNG(img,ball_marker_img,pos=(objects[0][0]-int(ball_w*0.2),objects[0][1]-int(ball_w*0.2)))
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img,(self.SCREEN_WIDTH,self.SCREEN_HEIGHT),cv2.INTER_AREA)
                img = np.rot90(img)
                img = cv2.flip(img,0)
                
                img = pygame.surfarray.make_surface(img)
                self.screen.blit(img, (0, 0))
            
                text = self.font.render(selected_drill, True, (255, 255, 255))
                self.screen.blit(text, (10, 10))
                pygame.display.update()

                if time.time() - start_time > 35:
                    self.show_image = False
                    self.selected_drill = None
                    cap.release()
                    break
    

    def run(self):
        self.running = True
        while self.running:
            self.handle_events()
            # self.screen.fill(self.WHITE)
            self.screen.blit(self.background_image, (0, 0))
            if not self.selected_drill:
                self.draw_buttons()
                pygame.display.update()
                self.clock.tick(60)
            else:
                self.display_live_camera_feed(self.selected_drill)  # Pass the selected drill name
                    
        

# Create an instance of the BasketballGameMenu class
menu = BasketballGameMenu(drills,camera_idx=1)

# Run the game
menu.run()
pygame.quit()
