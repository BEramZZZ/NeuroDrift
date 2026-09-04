import math


class Car:
    def __init__(self, x, y, angle=0.0):
        self.x = x
        self.y = y
        self.angle = angle          # degrees, 0 = facing right (pygame convention)
        self.speed = 0.0
        self.alive = True

        # engine / drag physics
        self.max_speed = 7.0
        self.max_reverse_speed = 2.5
        self.acceleration = 0.15
        self.friction = 0.03        # passive drag, always slowing the car slightly
        self.brake_power = 0.3      # extra deceleration when actively braking

        # steering physics (bicycle model): the car has a wheelbase, and its
        # turning radius depends on both the steering angle AND current speed -
        # you can't turn at all while stationary, and sharp turns at high speed
        # take a wider arc, just like a real car.
        self.wheelbase = 34.0            # distance between front/rear axle (pixels)
        self.max_steer_angle = 28.0      # degrees, how far the front wheels can turn
        self.steer_angle = 0.0           # current actual wheel angle (eases toward target)
        self.steer_response = 0.15       # how quickly the wheel turns toward the input (0-1)

        self.width = 20
        self.height = 40

    def move(self, steer: float, throttle: float):
        """
        steer:    -1.0 (left) to 1.0 (right) - steering INPUT, not the wheel angle itself
        throttle: -1.0 (brake/reverse) to 1.0 (full gas)
        """
        if not self.alive:
            return

        steer = max(-1.0, min(1.0, steer))
        throttle = max(-1.0, min(1.0, throttle))

        # --- speed update ---
        if throttle > 0:
            self.speed += throttle * self.acceleration
        elif throttle < 0:
            if self.speed > 0:
                # braking while moving forward: stronger deceleration than plain drag
                self.speed += throttle * self.brake_power
            else:
                self.speed += throttle * self.acceleration

        self.speed = max(-self.max_reverse_speed, min(self.max_speed, self.speed))

        # passive drag/friction, always nudging speed toward 0
        if self.speed > 0:
            self.speed = max(0.0, self.speed - self.friction)
        elif self.speed < 0:
            self.speed = min(0.0, self.speed + self.friction)

        # --- steering wheel eases toward the requested angle instead of snapping ---
        target_steer_angle = steer * self.max_steer_angle
        self.steer_angle += (target_steer_angle - self.steer_angle) * self.steer_response

        # --- bicycle model turning: angle change depends on speed AND wheel angle ---
        # a stationary car (speed=0) cannot turn at all, and turning is proportional
        # to how far the car actually travels this frame, not a flat per-frame amount
        if abs(self.speed) > 0.02:
            steer_rad = math.radians(self.steer_angle)
            angular_velocity_rad = (self.speed / self.wheelbase) * math.tan(steer_rad)
            self.angle += math.degrees(angular_velocity_rad)

        # move forward along current heading
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed

    def to_world(self, local_points):
        """Rotates and translates a list of (x, y) points defined in the car's
        own local space (forward = +x, sideways = +y) into world coordinates.
        Reused for both the collision rectangle and the visual car shape."""
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        world_points = []
        for lx, ly in local_points:
            rx = lx * cos_a - ly * sin_a
            ry = lx * sin_a + ly * cos_a
            world_points.append((self.x + rx, self.y + ry))
        return world_points

    def get_corners(self):
        """Returns the 4 corner points of the car's collision rectangle, rotated
        by self.angle. self.height is the front-to-back length (forward axis),
        self.width is the side-to-side width (sideways axis)."""
        half_length = self.height / 2
        half_width = self.width / 2

        local_corners = [
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width),
            (-half_length, -half_width),
        ]
        return self.to_world(local_corners)

    def kill(self):
        self.alive = False