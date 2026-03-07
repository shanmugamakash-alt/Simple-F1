import arcade
import numpy as np
import pandas as pd
import fastf1
import os

# Enable caching to speed up re-loading
if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache')

STATE_MENU, STATE_LOADING, STATE_RACING = 0, 1, 2

class RealisticF1Sim(arcade.Window):
    def __init__(self):
        super().__init__(1400, 900, "Simple F1", resizable=True)
        arcade.set_background_color(arcade.color.BLACK)
        
        self.state = STATE_MENU
        self.elapsed_time = 0.0
        self.speed_multiplier = 1.0
        
        # Menu Inputs
        self.input_year = "2023"
        self.input_event = "Silverstone"
        self.input_drivers = "20"
        self.input_mode = "One Lap" 
        self.active_field = 0
        
        self.driver_data = []
        self.track_line = []
        self.max_session_time = 0
        
        self.menu_title = arcade.Text("Simple F1", 700, 750, arcade.color.GOLD, 32, anchor_x="center")
        self.loading_text = arcade.Text("LOADING ASSETS...\n(Grand Prix mode may take 1-2 minutes)", 700, 450, arcade.color.WHITE, 20, anchor_x="center", multiline=True, width=800, align="center")

    def hex_to_color(self, hex_str):
        if not hex_str or hex_str == "": return arcade.color.WHITE
        try:
            h = hex_str.lstrip('#')
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        except:
            return arcade.color.WHITE

    def load_data_logic(self):
        try:
            clean_count = "".join(filter(str.isdigit, self.input_drivers))
            count = int(clean_count) if clean_count else 5
            
            session_type = 'R' if self.input_mode == "Grand Prix" else 'Q'
            session = fastf1.get_session(int(self.input_year), self.input_event, session_type)
            session.load(telemetry=True)
            
            track_angle = session.get_circuit_info().rotation / 180 * np.pi
            
            results = session.results.iloc[:count]
            drivers_to_load = results['Abbreviation'].tolist()
            
            # --- Load Track Layout ---
            ref_lap = session.laps.pick_fastest()
            if ref_lap is None or pd.isna(ref_lap['LapTime']):
                for drv in drivers_to_load:
                    temp_laps = session.laps.pick_drivers(drv)
                    if not temp_laps.empty:
                        ref_lap = temp_laps.pick_fastest()
                        if ref_lap is not None: break
            
            pos_data = ref_lap.get_pos_data().fill_missing()
            track_raw = pos_data.loc[:, ('X', 'Y')].to_numpy()
            rotated_track = self.rotate(track_raw, track_angle)
            
            self.final_scale = min((1100 * 0.8) / (rotated_track[:, 0].max() - rotated_track[:, 0].min()), 
                                   (900 * 0.8) / (rotated_track[:, 1].max() - rotated_track[:, 1].min()))
            self.offset_x = 850 - ((rotated_track[:, 0].min() + rotated_track[:, 0].max()) / 2) * self.final_scale
            self.offset_y = 450 - ((rotated_track[:, 1].min() + rotated_track[:, 1].max()) / 2) * self.final_scale
            
            self.track_line = [self.world_to_screen(p[0], p[1]) for p in rotated_track]
            
            if len(self.track_line) > 0:
                self.track_line.append(self.track_line[0])

            # --- Load Driver Telemetry ---
            global_min_time = float('inf')
            
            for drv in drivers_to_load:
                try:
                    laps = session.laps.pick_drivers(drv)
                    if laps.empty: continue
                    
                    if self.input_mode == "One Lap":
                        target_laps = laps.pick_fastest()
                        if target_laps is None: continue
                        telemetry = target_laps.get_pos_data().fill_missing()
                    else:
                        telemetry = laps.get_pos_data().fill_missing()
                    
                    pos = self.rotate(telemetry.loc[:, ('X', 'Y')].to_numpy(), track_angle)
                    times = telemetry['Time'].dt.total_seconds().to_numpy()
                    
                    if self.input_mode == "One Lap":
                        times = times - times[0] 
                    else:
                        global_min_time = min(global_min_time, times[0])
                        
                    drv_color = self.hex_to_color(session.get_driver(drv)['TeamColor'])
                    
                    self.driver_data.append({
                        'name': session.get_driver(drv)['Abbreviation'],
                        'times': times,
                        'positions': pos,
                        'color': drv_color,
                        'final_time': times[-1] - (times[0] if self.input_mode == "One Lap" else 0),
                        'label_text': arcade.Text(session.get_driver(drv)['Abbreviation'], 0, 0, drv_color, 10, bold=True)
                    })
                except Exception as e:
                    print(f"Skipping {drv}: {e}")

            if self.input_mode == "Grand Prix" and global_min_time != float('inf'):
                for driver in self.driver_data:
                    driver['times'] = driver['times'] - global_min_time
                    self.max_session_time = max(self.max_session_time, driver['times'][-1])
            else:
                for driver in self.driver_data:
                    self.max_session_time = max(self.max_session_time, driver['times'][-1])

            self.state = STATE_RACING
            self.speed_multiplier = 10.0 if self.input_mode == "Grand Prix" else 1.0 
            
        except Exception as e:
            print(f"Load Error: {e}")
            self.state = STATE_MENU

    def world_to_screen(self, x, y):
        return (x * self.final_scale + self.offset_x), (y * self.final_scale + self.offset_y)

    def rotate(self, xy, angle):
        rot_mat = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        return np.matmul(xy, rot_mat)

    def on_draw(self):
        self.clear()
        if self.state == STATE_MENU:
            self.menu_title.draw()
            fields = [f"Year: {self.input_year}", 
                      f"Track: {self.input_event}", 
                      f"Drivers: {self.input_drivers}",
                      f"Mode: < {self.input_mode} >"]
            
            for i, text in enumerate(fields):
                color = arcade.color.CYAN if i == self.active_field else arcade.color.WHITE
                arcade.draw_text(text, 700, 600 - (i*50), color, 18, anchor_x="center")
            
            arcade.draw_text("Use UP/DOWN to change fields. Left/Right to change mode. ENTER to start.", 700, 300, arcade.color.GRAY, 12, anchor_x="center")
                
        elif self.state == STATE_LOADING:
            self.loading_text.draw()
            
        elif self.state == STATE_RACING:
            arcade.draw_line_strip(self.track_line, arcade.color.DARK_SLATE_GRAY, 2)
            
            # Leaderboard Sidebar
            arcade.draw_lrbt_rectangle_filled(0, 280, 0, 900, (20, 20, 20, 220))
            leader_time = self.driver_data[0]['final_time'] if self.driver_data else 0
            
            for i, driver in enumerate(self.driver_data):
                y_tower = 830 - (i * 38)
                
                # Check if the driver is actively outputting telemetry at this exact second
                is_active = driver['times'][0] <= self.elapsed_time <= driver['times'][-1]
                
                # UI dims for inactive drivers
                text_color = arcade.color.WHITE if is_active else arcade.color.GRAY
                strip_color = driver['color'] if is_active else (60, 60, 60)
                
                arcade.draw_rect_filled(
                    arcade.LRBT(left=43, right=47, bottom=y_tower - 5, top=y_tower + 20),
                    strip_color
                )
                
                arcade.draw_text(f"{i+1} {driver['name']}", 60, y_tower, text_color, 12, bold=True)
                
                # Leaderboard Gaps
                if i == 0:
                    gap_text = "Leader" if self.input_mode == "Grand Prix" else "Interval"
                else:
                    gap = driver['final_time'] - leader_time
                    gap_text = f"+{gap:.3f}" if gap > 0 else "N/A"
                    
                arcade.draw_text(gap_text, 260, y_tower, arcade.color.GOLD if is_active else arcade.color.GRAY, 10, anchor_x="right")

                # ONLY draw the car on the track if it has active telemetry
                if is_active:
                    curr_x = np.interp(self.elapsed_time, driver['times'], driver['positions'][:, 0])
                    curr_y = np.interp(self.elapsed_time, driver['times'], driver['positions'][:, 1])
                    sx, sy = self.world_to_screen(curr_x, curr_y)
                    
                    arcade.draw_circle_filled(sx, sy, 6, driver['color'])
                    driver['label_text'].x, driver['label_text'].y = sx + 10, sy + 10
                    driver['label_text'].draw()
            
            mode_label = "Q - Fastest Laps" if self.input_mode == "One Lap" else "R - Grand Prix"
            arcade.draw_text(f"Mode: {mode_label} | Speed: {self.speed_multiplier}x | Time: {self.elapsed_time:.1f}s", 300, 20, arcade.color.WHITE, 12)

    def on_update(self, delta_time):
        if self.state == STATE_RACING:
            self.elapsed_time += delta_time * self.speed_multiplier
            if self.elapsed_time > self.max_session_time + 1:
                self.elapsed_time = 0

    def on_key_press(self, key, modifiers):
        if self.state == STATE_MENU:
            if key == arcade.key.DOWN or key == arcade.key.TAB:
                self.active_field = (self.active_field + 1) % 4
            elif key == arcade.key.UP:
                self.active_field = (self.active_field - 1) % 4
            elif key == arcade.key.LEFT or key == arcade.key.RIGHT:
                if self.active_field == 3:
                    self.input_mode = "Grand Prix" if self.input_mode == "One Lap" else "One Lap"
            elif key == arcade.key.ENTER:
                self.state = STATE_LOADING
                self.on_draw() 
                arcade.schedule_once(lambda dt: self.load_data_logic(), 0.1)
            elif key == arcade.key.BACKSPACE:
                if self.active_field == 0: self.input_year = self.input_year[:-1]
                elif self.active_field == 1: self.input_event = self.input_event[:-1]
                elif self.active_field == 2: self.input_drivers = self.input_drivers[:-1]
            else:
                try:
                    char = chr(key)
                    if char.isalnum() or char == " ":
                        if self.active_field == 0: self.input_year += char
                        elif self.active_field == 1: self.input_event += char
                        elif self.active_field == 2: self.input_drivers += char
                except: pass
        elif self.state == STATE_RACING:
            if key == arcade.key.UP: self.speed_multiplier += (0.5 if self.input_mode == "One Lap" else 5.0)
            elif key == arcade.key.DOWN: self.speed_multiplier = max(0.1, self.speed_multiplier - (0.5 if self.input_mode == "One Lap" else 5.0))
            elif key == arcade.key.SPACE: self.speed_multiplier = 0 if self.speed_multiplier > 0 else 1.0

if __name__ == "__main__":
    RealisticF1Sim()
    arcade.run()