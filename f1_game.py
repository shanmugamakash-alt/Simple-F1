import pygame
import numpy as np
import fastf1
from pygame.locals import *
import sys


class F1Game:
    def __init__(self):
        # Pygame setup
        pygame.init()
        self.WIDTH = 1400
        self.HEIGHT = 900
        self.display = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("F1 Track Visualization - Silverstone 2023 Qualifying")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 18)
        
        # Game state
        self.running = True
        self.playing = False
        self.current_frame = 0
        self.speed_multiplier = 1.0
        
        # Load F1 data
        print("Loading F1 data...")
        self.load_data()
        
        # UI buttons
        self.buttons = {
            'play': pygame.Rect(50, 50, 100, 40),
            'pause': pygame.Rect(160, 50, 100, 40),
            'reset': pygame.Rect(270, 50, 100, 40),
            'speed_up': pygame.Rect(400, 50, 80, 40),
            'speed_down': pygame.Rect(490, 50, 80, 40),
        }
    
    def load_data(self):
        """Load F1 session and driver data"""
        session = fastf1.get_session(2023, 'Silverstone', 'Q')
        session.load()
        
        self.circuit_info = session.get_circuit_info()
        self.drivers = session.drivers[:5]
        
        # Rotation angle
        track_angle = self.circuit_info.rotation / 180 * np.pi
        
        # Load driver data
        self.driver_data = []
        for driver in self.drivers:
            try:
                driver_laps = session.laps.pick_drivers(driver)
                if len(driver_laps) > 0:
                    fastest_lap = driver_laps.pick_fastest()
                    driver_pos = fastest_lap.get_pos_data()
                    positions = driver_pos.loc[:, ('X', 'Y')].to_numpy()
                    rotated = self.rotate(positions, angle=track_angle)
                    
                    driver_name = session.get_driver(driver)['Abbreviation']
                    
                    # Interpolate for smooth animation
                    smoothed = self.interpolate_smooth(rotated, factor=8)
                    
                    self.driver_data.append({
                        'name': driver_name,
                        'positions': smoothed,
                        'color': self.get_driver_color(len(self.driver_data))
                    })
                    print(f"Loaded {driver_name}")
            except Exception as e:
                print(f"Error loading driver: {e}")
        
        # Load track
        track = session.laps.pick_fastest().get_pos_data().loc[:, ('X', 'Y')].to_numpy()
        self.track = self.rotate(track, angle=track_angle)
        
        # Load corners
        self.corners = []
        offset_vector = np.array([500, 0])
        for _, corner in self.circuit_info.corners.iterrows():
            offset_angle = corner['Angle'] / 180 * np.pi
            offset_x, offset_y = self.rotate(offset_vector, angle=offset_angle)
            text_x = corner['X'] + offset_x
            text_y = corner['Y'] + offset_y
            text_x, text_y = self.rotate([text_x, text_y], angle=track_angle)
            track_x, track_y = self.rotate([corner['X'], corner['Y']], angle=track_angle)
            
            self.corners.append({
                'label': f"{corner['Number']}{corner['Letter']}",
                'text_pos': (text_x, text_y),
                'track_pos': (track_x, track_y)
            })
        
        self.max_frames = max(len(d['positions']) for d in self.driver_data)
        print(f"Max frames: {self.max_frames}")
    
    @staticmethod
    def rotate(xy, *, angle):
        """Rotate 2D points"""
        rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                            [-np.sin(angle), np.cos(angle)]])
        return np.matmul(xy, rot_mat)
    
    @staticmethod
    def interpolate_smooth(positions, factor=8):
        """Interpolate positions for smooth animation"""
        x = positions[:, 0]
        y = positions[:, 1]
        t = np.linspace(0, len(positions) - 1, len(positions) * factor)
        x_smooth = np.interp(t, np.arange(len(positions)), x)
        y_smooth = np.interp(t, np.arange(len(positions)), y)
        return np.column_stack([x_smooth, y_smooth])
    
    @staticmethod
    def get_driver_color(index):
        """Get color for driver"""
        colors = [
            (255, 0, 0),      # Red
            (255, 165, 0),    # Orange
            (0, 255, 0),      # Green
            (0, 165, 255),    # Light Blue
            (255, 0, 255),    # Magenta
        ]
        return colors[index % len(colors)]
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates"""
        # Scale and center
        scale = 2.5
        offset_x = self.WIDTH // 2
        offset_y = self.HEIGHT // 2
        
        screen_x = int(x * scale + offset_x)
        screen_y = int(y * scale + offset_y)
        
        return screen_x, screen_y
    
    def draw_background(self):
        """Draw background"""
        self.display.fill((20, 20, 20))
    
    def draw_track(self):
        """Draw the track"""
        points = [self.world_to_screen(p[0], p[1]) for p in self.track]
        if len(points) > 1:
            pygame.draw.lines(self.display, (255, 255, 255), points, 3)
    
    def draw_corners(self):
        """Draw corner labels"""
        for corner in self.corners:
            # Draw line
            pygame.draw.line(self.display, (100, 100, 100), 
                           self.world_to_screen(*corner['track_pos']),
                           self.world_to_screen(*corner['text_pos']), 1)
            
            # Draw circle
            pygame.draw.circle(self.display, (100, 100, 100),
                             self.world_to_screen(*corner['text_pos']), 12)
            
            # Draw text
            text = self.font_tiny.render(corner['label'], True, (255, 255, 255))
            text_rect = text.get_rect(center=self.world_to_screen(*corner['text_pos']))
            self.display.blit(text, text_rect)
    
    def draw_cars(self):
        """Draw car positions"""
        if self.current_frame >= self.max_frames:
            self.current_frame = 0
        
        for driver in self.driver_data:
            if self.current_frame < len(driver['positions']):
                pos = driver['positions'][self.current_frame]
                screen_x, screen_y = self.world_to_screen(pos[0], pos[1])
                
                # Draw car circle
                pygame.draw.circle(self.display, driver['color'], (screen_x, screen_y), 8)
                
                # Draw driver name
                name_text = self.font_tiny.render(driver['name'], True, driver['color'])
                self.display.blit(name_text, (screen_x + 12, screen_y - 10))
    
    def draw_ui(self):
        """Draw UI elements"""
        # Background
        pygame.draw.rect(self.display, (40, 40, 40), (10, 10, 600, 100))
        pygame.draw.rect(self.display, (80, 80, 80), (10, 10, 600, 100), 2)
        
        # Buttons
        button_color = (100, 150, 255)
        hover_color = (150, 200, 255)
        
        mouse_pos = pygame.mouse.get_pos()
        
        # Play button
        color = hover_color if self.buttons['play'].collidepoint(mouse_pos) else button_color
        pygame.draw.rect(self.display, color, self.buttons['play'])
        text = self.font_small.render("PLAY", True, (0, 0, 0))
        self.display.blit(text, (self.buttons['play'].centerx - 25, self.buttons['play'].centery - 12))
        
        # Pause button
        color = hover_color if self.buttons['pause'].collidepoint(mouse_pos) else button_color
        pygame.draw.rect(self.display, color, self.buttons['pause'])
        text = self.font_small.render("PAUSE", True, (0, 0, 0))
        self.display.blit(text, (self.buttons['pause'].centerx - 30, self.buttons['pause'].centery - 12))
        
        # Reset button
        color = hover_color if self.buttons['reset'].collidepoint(mouse_pos) else button_color
        pygame.draw.rect(self.display, color, self.buttons['reset'])
        text = self.font_small.render("RESET", True, (0, 0, 0))
        self.display.blit(text, (self.buttons['reset'].centerx - 30, self.buttons['reset'].centery - 12))
        
        # Speed up button
        color = hover_color if self.buttons['speed_up'].collidepoint(mouse_pos) else button_color
        pygame.draw.rect(self.display, color, self.buttons['speed_up'])
        text = self.font_small.render("SPEED+", True, (0, 0, 0))
        self.display.blit(text, (self.buttons['speed_up'].centerx - 35, self.buttons['speed_up'].centery - 12))
        
        # Speed down button
        color = hover_color if self.buttons['speed_down'].collidepoint(mouse_pos) else button_color
        pygame.draw.rect(self.display, color, self.buttons['speed_down'])
        text = self.font_small.render("SPEED-", True, (0, 0, 0))
        self.display.blit(text, (self.buttons['speed_down'].centerx - 35, self.buttons['speed_down'].centery - 12))
        
        # Status text
        status = "PLAYING" if self.playing else "PAUSED"
        status_text = self.font_small.render(f"Status: {status} | Speed: {self.speed_multiplier:.1f}x | Frame: {self.current_frame}/{self.max_frames}", 
                                            True, (200, 200, 200))
        self.display.blit(status_text, (50, 750))
        
        # Title
        title = self.font_large.render("F1 Silverstone 2023 - Qualifying", True, (255, 200, 0))
        self.display.blit(title, (self.WIDTH - 450, 50))
    
    def handle_events(self):
        """Handle user input"""
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            
            if event.type == MOUSEBUTTONDOWN:
                if self.buttons['play'].collidepoint(mouse_pos):
                    self.playing = True
                elif self.buttons['pause'].collidepoint(mouse_pos):
                    self.playing = False
                elif self.buttons['reset'].collidepoint(mouse_pos):
                    self.current_frame = 0
                    self.playing = False
                elif self.buttons['speed_up'].collidepoint(mouse_pos):
                    self.speed_multiplier = min(3.0, self.speed_multiplier + 0.5)
                elif self.buttons['speed_down'].collidepoint(mouse_pos):
                    self.speed_multiplier = max(0.5, self.speed_multiplier - 0.5)
            
            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    self.playing = not self.playing
                if event.key == K_r:
                    self.current_frame = 0
    
    def update(self):
        """Update game state"""
        if self.playing:
            self.current_frame += int(self.speed_multiplier)
            if self.current_frame >= self.max_frames:
                self.current_frame = 0
    
    def draw(self):
        """Draw everything"""
        self.draw_background()
        self.draw_track()
        self.draw_corners()
        self.draw_cars()
        self.draw_ui()
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = F1Game()
    game.run()
