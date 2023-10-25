import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pykalman import KalmanFilter



class dc_calculator():
	def __init__(self):
		self.prices = None
		self.time = None
		self.TMV_list = []
		self.T_list = []
		self.colors = []
		self.events = []



	def compute_dc_variables(self, threshold: float = 0.0001):
		"""

		Method to compute all relevant DC parameters.

		"""

		if self.prices is None:
			print('Please load the time series data first before proceeding with the DC parameters computation')
		else:
			self.TMV_list = []
			self.T_list = []
			self.colors = []
			self.events = []

			ext_point_n = self.prices[0]
			curr_event_max = self.prices[0]
			curr_event_min = self.prices[0]
			time_point_max = 0
			time_point_min = 0
			trend_status = 'up'
			T = 0

			for i in range(len(self.prices)):
				TMV = (self.prices[i] - ext_point_n) / (ext_point_n * threshold)
				self.TMV_list.append(TMV)
				self.T_list.append(T)
				T += 1

				if trend_status == 'up':
					self.colors.append('lime')
					self.events.append('Upward Overshoot')

					if self.prices[i] < ((1 - threshold) * curr_event_max):
						trend_status = 'down'
						curr_event_min = self.prices[i]

						ext_point_n = curr_event_max
						T = i - time_point_max

						num_points_change = i - time_point_max
						for j in range(1, num_points_change + 1):
							self.colors[-j] = 'red'
							self.events[-j] = 'Downward DCC'
					else:
						if self.prices[i] > curr_event_max:
							curr_event_max = self.prices[i]
							time_point_max = i
				else:
					self.colors.append('lightcoral')
					self.events.append('Downward Overshoot')

					if self.prices[i] > ((1 + threshold) * curr_event_min):
						trend_status = 'up'
						curr_event_max = self.prices[i]

						ext_point_n = curr_event_min			
						T = i - time_point_min

						num_points_change = i - time_point_min
						for j in range(1, num_points_change + 1):
							self.colors[-j] = 'green'
							self.events[-j] = 'Upward DCC'
					else:
						if self.prices[i] < curr_event_min:
							curr_event_min = self.prices[i]
							time_point_min = i

			self.colors = np.array(self.colors)

			print('DC variables computation has finished.')



	def generate_event_data(self, output_csv_name: str):
		"""

		Method to write the detected events for each 
		point in the time series data to a .csv file
		for external analysis. Again, feel free to 
		modify this method according to your needs.

		"""

		if isinstance(self.colors, list):
			print('Please load the time series data and compute DC variables first before attempting to generate the event data.')
		else:
			df = pd.DataFrame({'Time': self.time, 'Rate': self.prices, 'Event': np.array(self.events)})
			df.to_csv(output_csv_name + '.csv', index = False)

			print("The event data file '" + output_csv_name + ".csv' has been generated.")



	def generate_indicator_space_plot(self, title: str, output_plot_name = None):
		"""
	
		Method to generate the normalized TMV against normalized T indicator- 
		space plot after the DC parameters have been computed. This method
		currently uses the detected event class to colour the points, and it
		should be modified accordingly if one wishes to colour the points
		according to the detected regime as described in the book.

		"""

		if len(self.TMV_list) == 0:
			print('Please load the time series data and compute the DC variables before attempting to plot the indicator feature space.')
		else:
			TMV_array = np.array(self.TMV_list)
			T_array = np.array(self.T_list)
			norm_TMV = (TMV_array - np.min(TMV_array)) / (np.max(TMV_array) - np.min(TMV_array))
			norm_T = (T_array - np.min(T_array)) / (np.max(T_array) - np.min(T_array))

			fig, ax = plt.subplots()
			ax.scatter(norm_T, norm_TMV, c = self.colors, edgecolors = 'k')
			legend_elements = [plt.Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'green', markersize = 5, label = 'Upward DCC Event'),
							   plt.Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'lime', markersize = 5, label = 'Upward Overshoot Event'),
							   plt.Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'red', markersize = 5, label = 'Downward DCC Event'),
							   plt.Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'lightcoral', markersize = 5, label = 'Downward Overshoot Event')]
			ax.legend(handles = legend_elements, loc = 'upper right', fontsize = 'small')
			ax.set_xlim(np.min(norm_T) - 0.1, np.max(norm_T) + 0.1)
			ax.set_ylim(np.min(norm_TMV) - 0.1, np.max(norm_TMV) + 0.1)
			ax.set_title(title)
			ax.set_xlabel('Normalized T')
			ax.set_ylabel('Normalized TMV')
			if output_plot_name is not None and isinstance(output_plot_name, str):
				plt.savefig(output_plot_name + '.jpg')
			else:
				plt.show()

			print("The indicator feature space plot '" + title + "' has been generated.")



	def generate_original_time_series_plot(self, title: str, x_axis_label: str, y_axis_label: str, output_plot_name = None):
		"""

		Method to plot out the original time series data with no event
		annotation for visualization purpose. If output_plot_name is 
		supplied, then it is assumed that the plot should be written to
		file, and the corresponding plot .jpg file will be created.

		"""

		if self.prices is None:
			print('Please load the time series data first before plotting the original time series data.')
		else:
			fig2, ax2 = plt.subplots()
			ax2.ticklabel_format(style = 'plain', axis = 'y', useOffset = False)
			for i in range(len(self.prices)):
				ax2.plot(self.time[i : i + 2], self.prices[i : i + 2], color = 'black')
			ax2.set_xlim(0, len(self.prices) - 1)
			ax2.set_ylim(self.prices.min() * 0.9999, self.prices.max() * 1.0001)
			ax2.set_title(title)
			ax2.set_xlabel(x_axis_label)
			ax2.set_ylabel(y_axis_label)
			if output_plot_name is not None and isinstance(output_plot_name, str):
				plt.savefig(output_plot_name + '.jpg')
			else:
				plt.show()
			
			print("The original time series plot '" + title + "' has been generated.")



	def generate_time_series_animation(self, title: str, x_axis_label: str, y_axis_label: str, fps: int = 60, output_gif_name = None):
		"""
		
		Method to generate the event-annotated time series animation GIF after the DC parameters have been computed.

		"""
		if isinstance(self.colors, list):
			print('Please load the time series data and compute DC variables first before attempting to generate the time series animation.')
		else:
			green_patch = plt.Line2D([0], [0], color = 'green', label = 'Upward DCC Event')
			lime_patch = plt.Line2D([0], [0], color = 'lime', label = 'Upward Overshoot Event')
			red_patch = plt.Line2D([0], [0], color = 'red', label = 'Downward DCC Event')
			lightcoral_patch = plt.Line2D([0], [0], color = 'lightcoral', label = 'Downward Overshoot Event')

			fig, ax = plt.subplots()
			ax.ticklabel_format(style = 'plain', axis = 'y', useOffset = False)
			lines = [ax.plot([], [], color = color)[0] for color in self.colors]
			ax.set_xlim(0, len(self.prices) - 1)
			ax.set_ylim(self.prices.min() * 0.9999, self.prices.max() * 1.0001)
			ax.set_title(title)
			ax.set_xlabel(x_axis_label)
			ax.set_ylabel(y_axis_label)
			ax.legend(handles = [green_patch, lime_patch, red_patch, lightcoral_patch], loc = 'upper right', fontsize = 'small')



			def init():
				for line in lines:
					line.set_data([], [])
				return lines



			def update(i):
				if i == 0:
					return lines
				else:
					lines[i - 1].set_data(self.time[i - 1 : i + 1], self.prices[i - 1 : i + 1])
				return lines



			ani = animation.FuncAnimation(fig, update, frames = len(self.prices), init_func = init, blit = True)
			if output_gif_name is not None and isinstance(output_gif_name, str):
				ani.save(output_gif_name + '.gif', writer = 'pillow', fps = fps)
			else:
				plt.show()

			print("The animation '" + title + "' has been generated.")



	def generate_time_series_plot(self, title: str, x_axis_label: str, y_axis_label: str, output_plot_name = None):
		"""
		
		Method to generate the event-annotated time series plot after the DC parameters have been computed.

		"""

		if isinstance(self.colors, list):
			print('Please load the time series data and compute DC variables first before attempting to generate the time series plot.')
		else:
			green_patch = plt.Line2D([0], [0], color = 'green', label = 'Upward DCC Event')
			lime_patch = plt.Line2D([0], [0], color = 'lime', label = 'Upward Overshoot Event')
			red_patch = plt.Line2D([0], [0], color = 'red', label = 'Downward DCC Event')
			lightcoral_patch = plt.Line2D([0], [0], color = 'lightcoral', label = 'Downward Overshoot Event')

			fig1, ax1 = plt.subplots()
			ax1.ticklabel_format(style = 'plain', axis = 'y', useOffset = False)
			for i, color in enumerate(self.colors):
				ax1.plot(self.time[i : i + 2], self.prices[i : i + 2], color = color)
			ax1.set_xlim(0, len(self.prices) - 1)
			ax1.set_ylim(self.prices.min() * 0.9999, self.prices.max() * 1.0001)
			ax1.set_title(title)
			ax1.set_xlabel(x_axis_label)
			ax1.set_ylabel(y_axis_label)
			ax1.legend(handles = [green_patch, lime_patch, red_patch, lightcoral_patch], loc = 'upper right', fontsize = 'small')
			if output_plot_name is not None and isinstance(output_plot_name, str):
				plt.savefig(output_plot_name + '.jpg')
			else:
				plt.show()

			print("The plot '" + title + "' has been generated.")



	def load_time_series_data_from_file(self, file: str, data_point_limit = None, kalman_filter: bool = False):
		"""

		Method to load the time series data from a file. Please modify this method to suit your input data format. 
		The default method assumes that the data is stored in the format of a .csv file of which the columns
		are similar to that of the historical data provided by TrueFX (visit 
		https://www.truefx.com/truefx-historical-downloads/ for more details). In addition, this method also 
		provides an option to use the kalman filter on the time series data to smoothen it first before proceeding
		with the DC parameters computation

		"""
		try:
			df = pd.read_csv(file, header = None, names = ['Currency', 'Date', 'Short', 'Long'])
			self.prices = df['Long'].to_numpy()

			if data_point_limit is not None and data_point_limit < len(self.prices):
				self.prices = self.prices[:data_point_limit]

			if kalman_filter:
				kf = KalmanFilter(initial_state_mean = 0, n_dim_obs = 1)
				kf = kf.em(self.prices, n_iter = 10)
				(self.prices, _) = kf.smooth(self.prices)

			self.time = np.arange(len(self.prices))

			print("The time series data file '" + file + "' is loaded and ready for processing.")
		except:
			print('Please input a valid file name, and ensure that the file contains the data in an appropriate format.')



if __name__ == '__main__':
	calc = dc_calculator()

	# Example use of the methods of the dc_calculator class object
	calc.load_time_series_data_from_file('Truncated EURUSD-2023-09.csv', data_point_limit = 1000)
	calc.generate_original_time_series_plot('EUR-USD September 2023 Time Series', 
											'Time', 'EUR-USD', 
											output_plot_name = 'EURUSD-2023-09 Time Series Plot'
										   )
	calc.compute_dc_variables()
	calc.generate_indicator_space_plot('EUR-USD September 2023 Indicator Feature Space', 
									   output_plot_name = 'EURUSD-2023-09 Indicator Feature Space Plot'
									  )
	calc.generate_event_data('EURUSD-2023-09 Event Data')
	calc.generate_time_series_plot('Annotated EUR-USD September 2023 Time Series', 
								    'Time', 'EUR-USD', 
								    output_plot_name = 'Annotated EURUSD-2023-09 Plot'
								   )
	calc.generate_time_series_animation('Annotated EUR-USD September 2023 Time Series Animation', 
								   		 'Time', 'EUR-USD', 
								   		 output_gif_name = 'Annotated EURUSD-2023-09 Animation'
								   		)