#pragma once
#include <cmath>
#include <stdexcept>
#include <random>
#include <fstream>
//#include <iostream>
#include <string>
using namespace std;

#define MIN_ALPHA 0.1
#define MAX_ALPHA 0.3
#define MAX_ERR 0.1

class NModel
{
private:
	double** neurons;
	double*** weights;
	double* expected;
	double** delta;
	double alpha = 0.0;
	int input_size;
	int output_size;
	int inner_count;
	int* inner_size;
	int* model_size;
	typedef double (*activator)(double);
	activator* activators;

	void count_neural_net(void);
	void clear_neural_net(void);
	void recal_alpha(void);
	void adj_weight(void);
	
public:
	NModel(int input_size = 3,
		int output_size = 3,
		int inner_count = 1,
		int* inner_size = new int[1] {3}) :
		input_size(input_size),
		output_size(output_size),
		inner_count(inner_count),
		inner_size(inner_size) 
	{

		model_size = new int[inner_count + 2];
		model_size[0] = input_size;
		for (size_t i = 0; i < inner_count; i++)
		{
			model_size[i + 1] = inner_size[i];
		}
		model_size[inner_count + 1] = output_size;

		neurons = new double* [inner_count + 2];
		delta   = new double* [inner_count + 1];

		neurons[0] = new double[input_size];
		for (int i = 0; i < inner_count; i++)
		{
			neurons[i + 1] = new double[inner_size[i]];
			delta[i] = new double[inner_size[i]];
		}
		neurons[inner_count + 1] = new double[output_size];
		delta[inner_count] = new double[output_size];

		
		weights = new double** [inner_count + 1];

		weights[inner_count] = new double* [output_size];

		for (int i = 0; i < inner_count; i++)
		{
			weights[i] = new double* [inner_size[i]];
		}

		for (int i = 0; i < inner_size[0]; i++)
		{
			weights[0][i] = new double[input_size];
		}

		for (int i = 1; i < inner_count; i++)
		{
			for (int j = 0; j < inner_size[i]; j++)
			{
				weights[i][j] = new double[inner_size[i - 1]];
			}
		}
		for (int i = 0; i < output_size; i++)
		{
			weights[inner_count][i] = new double[inner_size[inner_count - 1]];
		}


		random_device device;
		mt19937 generator(device());
		uniform_real_distribution<float> distr(0, 0.3);
		

		//gen rand for 1 layer
		for (int i = 0; i < inner_size[0]; i++)
		{
			for (int j = 0; j < input_size; j++)
			{
				weights[0][i][j] = distr(generator);
			}
		}
		//gen rand for inner layers

		for (int i = 1; i < inner_count-1; i++)
		{
			for (int j = 0; j < inner_size[i]; j++)
			{
				for (int k = 0; k < inner_size[i-1]; k++)
				{
					weights[i][j][k] = distr(generator);
				}
			}

		}



		//gen rand for last layer

		for (int i = 0; i < output_size; i++)
		{
			for (int j = 0; j < inner_size[inner_count-1]; j++)
			{
				weights[inner_count][i][j] = distr(generator);
			}
		}


		expected = new double[output_size];
		activators = new activator[inner_count+1];
	};
	int  get_input_size() { return input_size; };
	int  get_output_size() { return output_size; };
	void load(const char* file_path);
	void save(const char* file_path);
	void set_input(int size, double* arr);
	void set_expected(int size, double* arr);
	void set_activator(int layer, activator func);
	void train(void);
	double err(void);
	void _res(int& size, double* arr);
	size_t apply(void);
};
	