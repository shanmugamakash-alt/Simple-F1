import tkinter as tk
from tkinter import Canvas, Button, Label, Frame
import numpy as np
import fastf1
from PIL import Image, ImageDraw, ImageTk

class F1App:
    def __init__(self, root):
        self.root = root
        self.root.title("F1 Silverstone 2023 - Qualifying")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a1a')
        
        self.canvas_width = 1200
        self.canvas_height = 750
        self.current_frame = 0
        self.max_frames = 0
        self.playing = False
        self.speed_multiplier = 1.0
        self.driver_data = []
        self.track = None
        self.corners = []
        self.show_controls = False  # Toggle for UI visibility
        
        print("Loading F1 data...")
        self.load_data()
        self.setup_ui()
        self.start_animation()
    
    def load_data(self):
        """Load F1 session and driver data"""
        session = fastf1.get_session(2023, 'Silverstone', 'Q')
        session.load()
        
        circuit_info = session.get_circuit_info()
        drivers = session.drivers[:5]
        track_angle = circuit_info.rotation / 180 * np.pi
        colors_rgb = [(255, 0, 0), (255, 165, 0), (0, 255, 0), (0, 165, 255), (255, 0, 255)]
        
        for idx, driver in enumerate(drivers):
            try:
                driver_laps = session.laps.pick_drivers(driver)
                if len(driver_laps) > 0:
                    fastest_lap = driver_laps.pick_fastest()
                    driver_pos = fastest_lap.get_pos_data()
                    positions = driver_pos.loc[:, ('X', 'Y')].to_numpy()
                    rotated = self.rotate(positions, angle=track_angle)
                    smoothed = self.interpolate_smooth(rotated, factor=8)
                    
                    driver_name = session.get_driver(driver)['Abbreviation']
                    self.driver_data.append({
                        'name': driver_name,
                        'positions': smoothed,
                        'color': colors_rgb[idx]
                    })
                    print(f"  ✓ {driver_name}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        # Load track
        track = session.laps.pick_fastest().get_pos_data().loc[:, ('X', 'Y')].to_numpy()
        self.track = self.rotate(track, angle=track_angle)
        
        # Load corners
        offset_vector = np.array([500, 0])
        for _, corner in circuit_info.corners.iterrows():
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
        print(f"\nLoaded! {len(self.driver_data)} drivers, {self.max_frames} frames\n")
    
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
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen"""
        # Get bounds of track to center it properly
        if not hasattr(self, '_bounds_calculated'):
            track_x = self.track[:, 0]
            track_y = self.track[:, 1]
            self.min_x = np.min(track_x)
            self.max_x = np.max(track_x)
            self.min_y = np.min(track_y)
            self.max_y = np.max(track_y)
            self._bounds_calculated = True
        
        # Normalize to 0-1
        norm_x = (x - self.min_x) / (self.max_x - self.min_x)
        norm_y = (y - self.min_y) / (self.max_y - self.min_y)
        
        # Scale to canvas with padding, and flip Y axis
        padding = 50
        scale_x = self.canvas_width - (2 * padding)
        scale_y = self.canvas_height - (2 * padding)
        
        screen_x = int(padding + norm_x * scale_x)
        screen_y = int(padding + (1 - norm_y) * scale_y)  # Flip Y
        
        return screen_x, screen_y
    
    def setup_ui(self):
        """Setup UI elements"""
        # Main frame
        self.main_frame = Frame(self.root, bg='#1a1a1a')
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title (always visible)
        title = Label(self.main_frame, text="F1 SILVERSTONE 2023 - QUALIFYING", 
                     font=('Arial', 18, 'bold'), bg='#1a1a1a', fg='#FFD700')
        title.pack(pady=10)
        
        # Canvas
        canvas_frame = Frame(self.main_frame, bg='#1a1a1a')
        canvas_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        self.canvas = Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height,
                            bg='#141414', highlightthickness=2, highlightbackground='#FFD700')
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.photo_image = None
        
        # Control panel (hidden by default)
        self.control_panel = Frame(self.main_frame, bg='#1a1a1a')
        self.control_panel.pack(pady=10)
        
        btn_style = {'font': ('Arial', 11, 'bold'), 'width': 12, 'bg': '#FFD700', 'fg': '#000000'}
        
        button_frame = Frame(self.control_panel, bg='#1a1a1a')
        button_frame.pack()
        
        Button(button_frame, text="▶ PLAY", command=self.play, **btn_style).grid(row=0, column=0, padx=5)
        Button(button_frame, text="⏸ PAUSE", command=self.pause, **btn_style).grid(row=0, column=1, padx=5)
        Button(button_frame, text="⏹ RESET", command=self.reset, **btn_style).grid(row=0, column=2, padx=5)
        Button(button_frame, text="⬆ SPEED+", command=self.speed_up, **btn_style).grid(row=0, column=3, padx=5)
        Button(button_frame, text="⬇ SPEED-", command=self.speed_down, **btn_style).grid(row=0, column=4, padx=5)
        
        # Status
        self.status_label = Label(self.control_panel, text="Status: PAUSED | Speed: 1.0x | Frame: 0/0",
                                 font=('Arial', 11), bg='#1a1a1a', fg='#FFFFFF')
        self.status_label.pack(pady=5)
        
        # Legend
        legend_frame = Frame(self.control_panel, bg='#1a1a1a')
        legend_frame.pack(pady=5)
        
        Label(legend_frame, text="DRIVERS:", font=('Arial', 10, 'bold'), 
              bg='#1a1a1a', fg='#FFFFFF').pack(side=tk.LEFT, padx=10)
        
        for driver in self.driver_data:
            color_hex = f'#{driver["color"][0]:02x}{driver["color"][1]:02x}{driver["color"][2]:02x}'
            Label(legend_frame, text=f'● {driver["name"]}',
                  font=('Arial', 10), bg='#1a1a1a', fg=color_hex).pack(side=tk.LEFT, padx=5)
        
        # Toggle button in corner
        self.toggle_btn = Button(self.root, text="☰", command=self.toggle_controls, 
                                font=('Arial', 10, 'bold'), width=3, height=1,
                                bg='#FFD700', fg='#000000')
        self.toggle_btn.place(x=10, y=10)
        
        # Hide controls initially
        self.control_panel.pack_forget()
    
    def on_canvas_click(self, event):
        """Toggle controls on canvas click"""
        self.toggle_controls()
    
    def toggle_controls(self):
        """Show/hide control panel"""
        self.show_controls = not self.show_controls
        if self.show_controls:
            self.control_panel.pack(pady=10)
        else:
            self.control_panel.pack_forget()
    
    def draw_frame(self):
        """Draw current frame"""
        img = Image.new('RGB', (self.canvas_width, self.canvas_height), color=(20, 20, 20))
        draw = ImageDraw.Draw(img)
        
        # Draw track
        track_points = [self.world_to_screen(p[0], p[1]) for p in self.track]
        if len(track_points) > 1:
            draw.line(track_points, fill=(255, 255, 255), width=4)
        
        # Draw corners
        for corner in self.corners:
            track_pos = self.world_to_screen(*corner['track_pos'])
            text_pos = self.world_to_screen(*corner['text_pos'])
            
            draw.line([track_pos, text_pos], fill=(100, 100, 100), width=1)
            draw.ellipse([text_pos[0]-12, text_pos[1]-12, text_pos[0]+12, text_pos[1]+12], 
                        outline=(100, 100, 100), width=1)
            draw.text((text_pos[0]-6, text_pos[1]-6), corner['label'], fill=(255, 255, 255))
        
        # Draw cars
        if self.current_frame < self.max_frames:
            for driver in self.driver_data:
                if self.current_frame < len(driver['positions']):
                    pos = driver['positions'][self.current_frame]
                    screen_x, screen_y = self.world_to_screen(pos[0], pos[1])
                    
                    draw.ellipse([screen_x-10, screen_y-10, screen_x+10, screen_y+10], 
                               fill=driver['color'])
                    draw.text((screen_x+12, screen_y-8), driver['name'], fill=driver['color'])
        
        self.photo_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW)
    
    def update_animation(self):
        """Update animation loop"""
        if self.playing:
            self.current_frame += int(self.speed_multiplier)
            if self.current_frame >= self.max_frames:
                self.current_frame = 0
        
        self.draw_frame()
        
        status = "PLAYING" if self.playing else "PAUSED"
        self.status_label.config(
            text=f"Status: {status} | Speed: {self.speed_multiplier:.1f}x | Frame: {self.current_frame}/{self.max_frames}"
        )
        
        self.root.after(16, self.update_animation)
    
    def start_animation(self):
        """Start animation loop"""
        self.update_animation()
    
    def play(self):
        self.playing = True
    
    def pause(self):
        self.playing = False
    
    def reset(self):
        self.current_frame = 0
        self.playing = False
    
    def speed_up(self):
        self.speed_multiplier = min(3.0, self.speed_multiplier + 0.5)
    
    def speed_down(self):
        self.speed_multiplier = max(0.5, self.speed_multiplier - 0.5)

if __name__ == '__main__':
    root = tk.Tk()
    app = F1App(root)
    print("🏁 F1 App Running!")
    print("   Controls: PLAY, PAUSE, RESET, SPEED+, SPEED-\n")
    root.mainloop()
