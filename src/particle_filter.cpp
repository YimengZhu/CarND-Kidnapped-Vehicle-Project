/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;

	normal_distribution<double> x_dist(x, std[0]);
	normal_distribution<double> y_dist(y, std[1]);
	normal_distribution<double> theta_dist(theta, std[2]);

	default_random_engine gen;

	for(int i = 0; i < num_particles; i++){
		Particle p;
		p.id = i;
		p.weight = 1.0;
		p.x = x_dist(gen);
		p.y = y_dist(gen);
		p.theta = theta_dist(gen);
		particles.push_back(p);
		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	for(auto& particle:particles) {
		double theta_pred, x_pred, y_pred;

		if (abs(yaw_rate) > 1e-4) {
			theta_pred = particle.theta + yaw_rate * delta_t;
			x_pred	   = particle.x + velocity / yaw_rate * (sin(theta_pred) - sin(particle.theta));
			y_pred	   = particle.y + velocity / yaw_rate * (cos(particle.theta) - cos(theta_pred));

		} else {
			theta_pred = particle.theta;
			x_pred	   = particle.x + velocity * delta_t * cos(particle.theta);
			y_pred	   = particle.y + velocity * delta_t * sin(particle.theta);
		}

		normal_distribution<double> x_dist(x_pred, std_pos[0]);
		normal_distribution<double> y_dist(y_pred, std_pos[1]);
		normal_distribution<double> theta_dist(theta_pred, std_pos[2]);

		particle.x = x_dist(gen);
		particle.y = y_dist(gen);
		particle.theta = theta_dist(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(auto& obs : observations) {

		double min_distance = numeric_limits<double>::max();

		for(auto pred:predicted){

			double distance = dist(obs.x, obs.y, pred.x, pred.y);

			if(distance < min_distance){
				min_distance = distance;
				obs.id = pred.id;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

 	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];

	// Iterate over all particles
	for (size_t i = 0; i < num_particles; ++i) {


		vector<LandmarkObs> valid_landmarks;
        for(auto& landmark: map_landmarks.landmark_list){
            double distance = dist(particles[i].x, particles[i].y, landmark.x_f, landmark.y_f);
            
            if(distance <= sensor_range){
                LandmarkObs landmark_vaild{landmark.id_i, landmark.x_f, landmark.y_f};
                valid_landmarks.push_back(landmark_vaild);
            }
        }


			// List all observations in map coordinates
		vector<LandmarkObs> map_landmarks;
        for(auto& obs : observations) {
            double x_map = cos(particles[i].theta) * obs.x - sin(particles[i].theta) * obs.y + particles[i].x;
            double y_map = sin(particles[i].theta) * obs.x + cos(particles[i].theta) * obs.y + particles[i].y;
            LandmarkObs obs_map{obs.id, x_map,y_map};
            map_landmarks.push_back(obs_map);
        }


		dataAssociation(valid_landmarks, map_landmarks);

		double weight = 1.0;

        double u_x, u_y;
        
        for(const auto& obs_map:valid_landmarks){
            for(const auto& landmark:map_landmarks){
                if(obs_map.id == landmark.id){
                    u_x = landmark.x;
                    u_y = landmark.y;
                    break;
                }
            }
            
        double prob = exp(-(pow(obs_map.x - u_x, 2) / (2 * sigma_x * sigma_x) +
                        pow(obs_map.y - u_y, 2) / (2 * sigma_y * sigma_y))) /
                        	(2 * M_PI * sigma_x * sigma_y);
            weight *= prob;
        }
        particles[i].weight = weight;

		}
    
    double weight_sum = 0.0;
    for(const auto& particle:particles) {
        weight_sum += particle.weight;
    }
    
    for(int i =0; i < num_particles; i++) {
        particles[i].weight /= weight_sum;
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	vector<Particle> resampled_particles;

	for(int i = 0; i < num_particles; i++) {
		discrete_distribution<int> index(weights.begin(), weights.end());
		resampled_particles.push_back(particles[index(gen)]);
	}

	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
