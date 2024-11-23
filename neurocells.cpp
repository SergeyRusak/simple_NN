// neurocells.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include "NModel.h"
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace std::filesystem;

#define LEARNING_BARRIER 0.001
#define MAX_BARRIER 3
#define MAX_ERR 0.1

double linear(double x) {
	return x;
}
double ReLu(double x) {
    return (x < 0) ? (0) : (x);
}
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

void train_model(NModel m, int max_epoch = 100, const path train_paths = "./input/train") {
    int input_size = m.get_input_size();
    int output_size = m.get_output_size();
    double* expected = new double[output_size];
    double* input = new double[input_size];
    double old_mean_err = 10;
    double count_barrier = 0;
    double mean_err;
    int epoch;
    for (epoch = 0; epoch < 100; epoch++)
    {
        mean_err = 0;
        int mean_count = 0;
        for (auto& train_path : directory_iterator(train_paths))
        {
            ifstream fin(train_path.path());
            fin >> expected[0] >> expected[1] >> expected[2];
            m.set_expected(3, expected);
            for (size_t i = 0; i < 7 * 7; i++)
                fin >> input[i];
            m.set_input(7 * 7, input);
            m.train();
            mean_err += m.err();
            mean_count++;
        }
        mean_err /= mean_count;
        if (abs(old_mean_err - mean_err) < LEARNING_BARRIER || mean_err > old_mean_err)
            count_barrier++;
        else
            count_barrier = 0;
        old_mean_err = mean_err;
        if (count_barrier == MAX_BARRIER && mean_err < MAX_ERR) {
            break;
        }
    }
    cout << "Train stopped on " << epoch << " epoch" << endl;
    cout << "Mear err on train is " << mean_err << endl;
}
void validate_model(NModel m, const path vaidate_paths = "./input/validate") {
    int mean_err = 0;
    int mean_count = 0;
    int input_size = m.get_input_size();
    int output_size = m.get_output_size();
    double* expected = new double[output_size];
    double* input = new double[input_size];
    for (auto& val_path : directory_iterator(vaidate_paths))
    {
        ifstream fin(val_path.path());
        fin >> expected[0] >> expected[1] >> expected[2];
        m.set_expected(3, expected);
        for (size_t i = 0; i < 7 * 7; i++)
            fin >> input[i];
        m.set_input(7 * 7, input);
        size_t classified = m.apply();
        if (!expected[classified]) {
            cout << "_______________________________________________" << endl;
            cout << "Error on image " << val_path.path() << endl;
            cout << "Expected: ";
            if (expected[0])
                cout << "Circle" << endl;
            else if (expected[1])
                cout << "Square" << endl;
            else
                cout << "Triangle" << endl;
            cout << "Actual: ";
            if (classified == 0)
                cout << "Circle" << endl;
            else if (classified == 1)
                cout << "Square" << endl;
            else
                cout << "Triangle" << endl;
            cout << "_______________________________________________" << endl;
        }
        mean_err += m.err();
        mean_count++;
    }
    mean_err /= mean_count;
    cout << "Mear err on validation is " << mean_err << endl;
}

int main()
{
	int input_size = 7 * 7;
	int output_size = 3;
	int inner_layers_size = 1;
	int* inner_size = new int[inner_layers_size]{ 14 };

	NModel m(input_size, output_size, inner_layers_size, inner_size);
	m.set_activator(1, sigmoid);
	m.set_activator(2, sigmoid);
    m.load("weight.txt");
    train_model(m);
    validate_model(m);
    m.save("weight.txt");
  

}


