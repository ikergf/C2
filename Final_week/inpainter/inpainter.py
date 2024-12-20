import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import laplace
from scipy.ndimage.filters import convolve
from tqdm import tqdm

class Inpainter():
    def __init__(self, image, mask, patch_size=9, plot_progress=True):
        self.image = image.astype('uint8')
        self.mask = mask.round().astype('uint8')
        self.patch_size = patch_size
        self.plot_progress = plot_progress
        # self.search_space = search_space # No aplica este parametro para el swarm optimization

        # Non initialized attributes
        self.working_image = None
        self.working_mask = None
        self.front = None
        self.confidence = None
        self.data = None
        self.priority = None

    def inpaint(self):
        """ Compute the new image and return it """

        self._validate_inputs()
        self._initialize_attributes()

        start_time = time.time()
        keep_going = True
        while keep_going:
            self._find_front()
            if self.plot_progress:
                self._plot_image()

            self._update_priority()

            target_pixel = self._find_highest_priority_pixel()
            # print(target_pixel[0])
            find_start_time = time.time()
            source_patch = self._find_source_patch(target_pixel)
            print('Time to find best: %f seconds'
                  % (time.time()-find_start_time))

            self._update_image(target_pixel, source_patch)

            keep_going = not self._finished()

        print('Took %f seconds to complete' % (time.time() - start_time))
        return self.working_image

    def _validate_inputs(self):
        if self.image.shape[:2] != self.mask.shape:
            raise AttributeError('mask and image must be of the same size')

    def _plot_image(self):
        height, width = self.working_mask.shape

        # Remove the target region from the image
        inverse_mask = 1 - self.working_mask
        rgb_inverse_mask = self._to_rgb(inverse_mask)
        image = self.working_image * rgb_inverse_mask

        # Fill the target borders with red
        image[:, :, 0] += self.front * 255

        # Fill the inside of the target region with white
        white_region = (self.working_mask - self.front) * 255
        rgb_white_region = self._to_rgb(white_region)
        image += rgb_white_region

        plt.clf()
        plt.imshow(image)
        plt.draw()
        plt.pause(0.001)  # TODO: check if this is necessary

    def _initialize_attributes(self):
        """ Initialize the non initialized attributes

        The confidence is initially the inverse of the mask, that is, the
        target region is 0 and source region is 1.

        The data starts with zero for all pixels.

        The working image and working mask start as copies of the original
        image and mask.
        """
        height, width = self.image.shape[:2]

        self.confidence = (1 - self.mask).astype(float)
        self.data = np.zeros([height, width])

        self.working_image = np.copy(self.image)
        self.working_mask = np.copy(self.mask)

    def _find_front(self):
        """ Find the front using laplacian on the mask

        The laplacian will give us the edges of the mask, it will be positive
        at the higher region (white) and negative at the lower region (black).
        We only want the the white region, which is inside the mask, so we
        filter the negative values.
        """
        self.front = (laplace(self.working_mask) > 0).astype('uint8')
        # TODO: check if scipy's laplace filter is faster than scikit's

    def _update_priority(self):
        self._update_confidence()
        self._update_data()
        self.priority = self.confidence * self.data * self.front

    def _update_confidence(self):
        new_confidence = np.copy(self.confidence)
        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            new_confidence[point[0], point[1]] = sum(sum(
                self._patch_data(self.confidence, patch)
            ))/self._patch_area(patch)

        self.confidence = new_confidence

    def _update_data(self):
        normal = self._calc_normal_matrix()
        gradient = self._calc_gradient_matrix()

        normal_gradient = normal*gradient
        self.data = np.sqrt(
            normal_gradient[:, :, 0]**2 + normal_gradient[:, :, 1]**2
        ) + 0.001  # To be sure to have a greater than 0 data

    def _calc_normal_matrix(self):
        x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

        x_normal = convolve(self.working_mask.astype(float), x_kernel)
        y_normal = convolve(self.working_mask.astype(float), y_kernel)
        normal = np.dstack((x_normal, y_normal))

        height, width = normal.shape[:2]
        norm = np.sqrt(y_normal**2 + x_normal**2) \
                 .reshape(height, width, 1) \
                 .repeat(2, axis=2)
        norm[norm == 0] = 1

        unit_normal = normal/norm
        return unit_normal

    def _calc_gradient_matrix(self):
        # TODO: find a better method to calc the gradient
        height, width = self.working_image.shape[:2]

        grey_image = rgb2gray(self.working_image)
        grey_image[self.working_mask == 1] = None

        gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
        max_gradient = np.zeros([height, width, 2])

        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            patch_y_gradient = self._patch_data(gradient[0], patch)
            patch_x_gradient = self._patch_data(gradient[1], patch)
            patch_gradient_val = self._patch_data(gradient_val, patch)

            patch_max_pos = np.unravel_index(
                patch_gradient_val.argmax(),
                patch_gradient_val.shape
            )

            max_gradient[point[0], point[1], 0] = \
                patch_y_gradient[patch_max_pos]
            max_gradient[point[0], point[1], 1] = \
                patch_x_gradient[patch_max_pos]

        return max_gradient

    def _find_highest_priority_pixel(self):
        point = np.unravel_index(self.priority.argmax(), self.priority.shape)
        return point

    def _find_source_patch(self, target_pixel):
        target_patch = self._get_patch(target_pixel)
        height, width = self.working_image.shape[:2]
        patch_height, patch_width = self._patch_shape(target_patch)

        half_patch_size = (self.patch_size-1)//2

        min_y = max(0, target_pixel[0] - half_patch_size - self.patch_size*self.search_space)
        max_y = min(target_pixel[0] + half_patch_size + self.patch_size*self.search_space, height-1)

        min_x = max(0, target_pixel[1] - half_patch_size - self.patch_size*self.search_space)
        max_x = min(target_pixel[1] + half_patch_size + self.patch_size*self.search_space, width-1)

        best_match = None
        best_match_difference = 0

        lab_image = rgb2lab(self.working_image)

        for y in range(min_y, max_y- patch_height + 1):
            for x in range(min_x, max_x- patch_width + 1):
                source_patch = [
                    [y, y + patch_height-1],
                    [x, x + patch_width-1]
                ]
                if self._patch_data(self.working_mask, source_patch) \
                   .sum() != 0:
                    continue

                difference = self._calc_patch_difference(
                    lab_image,
                    target_patch,
                    source_patch
                )

                if best_match is None or difference < best_match_difference:
                    best_match = source_patch
                    best_match_difference = difference
        return best_match

    def _update_image(self, target_pixel, source_patch):
        target_patch = self._get_patch(target_pixel)
        pixels_positions = np.argwhere(
            self._patch_data(
                self.working_mask,
                target_patch
            ) == 1
        ) + [target_patch[0][0], target_patch[1][0]]
        patch_confidence = self.confidence[target_pixel[0], target_pixel[1]]
        for point in pixels_positions:
            self.confidence[point[0], point[1]] = patch_confidence

        mask = self._patch_data(self.working_mask, target_patch)
        rgb_mask = self._to_rgb(mask)
        source_data = self._patch_data(self.working_image, source_patch)
        target_data = self._patch_data(self.working_image, target_patch)

        new_data = source_data*rgb_mask + target_data*(1-rgb_mask)

        self._copy_to_patch(
            self.working_image,
            target_patch,
            new_data
        )
        self._copy_to_patch(
            self.working_mask,
            target_patch,
            0
        )

    def _get_patch(self, point):
        half_patch_size = (self.patch_size-1)//2
        height, width = self.working_image.shape[:2]
        patch = [
            [
                max(0, point[0] - half_patch_size),
                min(point[0] + half_patch_size, height-1)
            ],
            [
                max(0, point[1] - half_patch_size),
                min(point[1] + half_patch_size, width-1)
            ]
        ]
        return patch

    def _calc_patch_difference(self, image, target_patch, source_patch):
        mask = 1 - self._patch_data(self.working_mask, target_patch)
        rgb_mask = self._to_rgb(mask)
        target_data = self._patch_data(
            image,
            target_patch
        ) * rgb_mask
        source_data = self._patch_data(
            image,
            source_patch
        ) * rgb_mask
        squared_distance = ((target_data - source_data)**2).sum()
        euclidean_distance = np.sqrt(
            (target_patch[0][0] - source_patch[0][0])**2 +
            (target_patch[1][0] - source_patch[1][0])**2
        )  # tie-breaker factor
        return squared_distance + euclidean_distance

    def _finished(self):
        height, width = self.working_image.shape[:2]
        remaining = self.working_mask.sum()
        total = height * width
        print('%d of %d completed' % (total-remaining, total))
        return remaining == 0

    @staticmethod
    def _patch_area(patch):
        return (1+patch[0][1]-patch[0][0]) * (1+patch[1][1]-patch[1][0])

    @staticmethod
    def _patch_shape(patch):
        return (1+patch[0][1]-patch[0][0]), (1+patch[1][1]-patch[1][0])

    @staticmethod
    def _patch_data(source, patch):
        return source[
            patch[0][0]:patch[0][1]+1,
            patch[1][0]:patch[1][1]+1
        ]

    @staticmethod
    def _copy_to_patch(dest, dest_patch, data):
        dest[
            dest_patch[0][0]:dest_patch[0][1]+1,
            dest_patch[1][0]:dest_patch[1][1]+1
        ] = data

    @staticmethod
    def _to_rgb(image):
        height, width = image.shape
        return image.reshape(height, width, 1).repeat(3, axis=2)
    
class PSOInpainter_2(Inpainter):
    def __init__(self, image, mask, patch_size=9, plot_progress=True, num_particles=50, w=0.5, c1=2.0, c2=2.0, iterations = 100):
        super().__init__(image, mask, patch_size, plot_progress)
        self.num_particles = num_particles
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.iterations = iterations

    def _find_source_patch(self, target_pixel):
        """ Use PSO to find the best matching source patch for the target patch. """
        target_patch = self._get_patch(target_pixel)
        lab_image = rgb2lab(self.working_image)
        patch_height, patch_width = self._patch_shape(target_patch)
        # Initialize particles randomly over valid source regions
        particles = self._initialize_particles()
        velocities = np.random.uniform(-3, 3, (self.num_particles, 2))  # Random velocities
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.num_particles, np.inf)
        global_best_position = None
        global_best_score = np.inf

        # Run PSO iterations
        for iteration in range(self.iterations):  # Set number of PSO iterations
            for i, particle in enumerate(particles):
                source_patch = self._create_patch_from_particle(particle, patch_height, patch_width)
                if not self._is_valid_source_patch(source_patch):
                    continue
                fitness = self._calc_patch_difference(lab_image, target_patch, source_patch)

                # Update personal best
                if fitness < personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = particle

                # Update global best
                if fitness < global_best_score:
                    global_best_score = fitness
                    global_best_position = particle

            # Update velocities and positions for particles
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]))
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], 0, [self.image.shape[0] - self.patch_size, self.image.shape[1] - self.patch_size])

        # Use the global best position as the selected source patch
        return self._create_patch_from_particle(global_best_position,patch_height, patch_width)

    # def _initialize_particles(self):
    #     """ Initialize particles randomly over valid source regions. """
    #     height, width = self.image.shape[:2]
    #     particles = np.random.randint(0, [height - self.patch_size-1, width - self.patch_size-1], (self.num_particles, 2))
    #     return particles

    def _initialize_particles(self):
        """Initialize particles in valid source regions only (regions where mask == 0)."""
        height, width = self.image.shape[:2]
        particles = []

        # Loop until we have enough particles
        while len(particles) < self.num_particles:
            y, x = np.random.randint(0, height - self.patch_size - 1), np.random.randint(0, width - self.patch_size - 1)
            source_patch = [[y, y + self.patch_size - 1], [x, x + self.patch_size - 1]]

            # Check if the sampled patch is in a valid (non-masked) region
            if self._is_valid_source_patch(source_patch):
                particles.append([y, x])

        return np.array(particles)

    def _create_patch_from_particle(self, particle, patch_height, patch_width):
        """ Create patch boundaries based on a particle's position. """
        y, x = particle
        return [[y, y + patch_height - 1], [x, x + patch_width - 1]]

    def _is_valid_source_patch(self, source_patch):
        """ Check if the source patch does not overlap the mask. """
        return self._patch_data(self.working_mask, source_patch).sum() == 0

# Example usage:
# image and mask would be your input image and the corresponding mask where the target region is
# inpainter = PSO_Inpainter(image, mask)
# inpainted_image = inpainter.inpaint()
