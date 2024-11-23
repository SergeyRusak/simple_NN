#include "NModel.h"




void NModel::count_neural_net(void)
{
    for (int layer = 0; layer <= inner_count; layer++)
    {
        for (int neuron = 0; neuron < model_size[layer+1]; neuron++)
        {
            for (int input = 0; input < model_size[layer]; input++)
            {
                neurons[layer + 1][neuron] += neurons[layer][input] * weights[layer][neuron][input];
            }
            neurons[layer + 1][neuron] = activators[layer](neurons[layer + 1][neuron]);
        }
    }


}

void NModel::clear_neural_net(void)
{
    for (int layer = 0; layer <= inner_count; layer++)
        for (int neuron = 0; neuron < model_size[layer+1]; neuron++)
            neurons[layer + 1][neuron] = 0;
}

void NModel::recal_alpha(void)
{
    double e = err();
    double rel_e = 2 * abs(e) / output_size;
    alpha = rel_e * (MAX_ALPHA - MIN_ALPHA) + MIN_ALPHA;
}

void NModel::adj_weight(void)
{
    for (int exp = 0; exp < output_size; exp++)
    {
        double t = expected[exp];
        double y = neurons[inner_count + 1][exp];
        delta[inner_count][exp] = y * (1 - y) * (t - y);
    }
    for (int layer = inner_count - 1; layer >= 0; layer--)
    {
        for (int input = 0; input < model_size[layer + 1]; input++)
        {
            double next_sum = 0;
            for (int next_neuron = 0; next_neuron < model_size[layer + 2]; next_neuron++)
            {
                next_sum += delta[layer + 1][next_neuron] * weights[layer + 1][next_neuron][input];
            }
            delta[layer][input] = neurons[layer + 1][input] * (1 - neurons[layer + 1][input]) * next_sum;
        }
    }
    for (int layer = 0; layer < inner_count + 1; layer++)
    {
        for (int neuron = 0; neuron < model_size[layer+1]; neuron++)
        {
            for (int input = 0; input < model_size[layer]; input++)
            {
               // cout << layer << ":" << neuron << ":" << input << " ->" << weights[layer][neuron][input] << endl;
                weights[layer][neuron][input] += alpha * delta[layer][neuron] * neurons[layer][input];
            }
        }
    }


}

void NModel::load(const char* file_path)
{
    ifstream load_weight(file_path);
    int inp_f, inn_f, out_f;
    int* inn_s;
    double outp;
    if (load_weight.good()) {
        load_weight >> inp_f >> inn_f;
        if (inp_f != input_size) throw invalid_argument("Wrong input size in file!");
        if (inn_f != inner_count) throw invalid_argument("Wrong inner size in file! (inner layers count not equal)");
        inn_s = new int[inn_f];
        int temp;
        for (size_t i = 0; i < inn_f; i++)
        {
            load_weight >> temp;
            if (temp != inner_size[i]) throw invalid_argument("Wrong inner layer size in file!");
        }
        load_weight >> out_f;
        if (out_f != output_size) throw invalid_argument("Wrong output size in file!");
    
        for (int layer = 0; layer < inner_count + 1; layer++)
        {
            for (int neuron = 0; neuron < model_size[layer + 1]; neuron++)
            {
                
                for (int input = 0; input < model_size[layer]; input++)
                {
                    load_weight >> weights[layer][neuron][input];
                }
                
            }
        }
    }
    


}

void NModel::save(const char* file_path)
{
    ofstream save_weight;
    save_weight.open(file_path);
    save_weight << input_size << endl;
    save_weight << inner_count<< " " << inner_size[0];
    for (size_t i = 1; i < inner_count; i++)
    {
        save_weight <<  " " << inner_size[i];
    }
    save_weight << endl;
    save_weight << output_size << endl;
    for (int layer = 0; layer < inner_count + 1; layer++)
    {
        for (int neuron = 0; neuron < model_size[layer + 1]; neuron++)
        {
            save_weight << to_string(weights[layer][neuron][0]);

            for (int input = 1; input < model_size[layer]; input++)
            {
                save_weight << " " << to_string(weights[layer][neuron][input]);
            }
            save_weight << endl;
        }
    }
    save_weight.close();

}

void NModel::set_input(int size, double* arr)
{
    if (size != input_size)
    {
        throw invalid_argument("Wrong input size!");
    }
    for (int i = 0; i < size; i++)
    {
        neurons[0][i] = arr[i];
    }

}

void NModel::set_expected(int size, double* arr)
{
    if (size != output_size)
    {
        throw invalid_argument("Wrong input size!");
    }
    for (int i = 0; i < size; i++)
    {
        expected[i] = arr[i];
    }
}

void NModel::set_activator(int layer, activator func)
{
    if (layer-1 <0 || layer-1 >inner_count)
    {
        throw invalid_argument("wrong layer index");
    }
    activators[layer - 1] = func;

}

void NModel::train(void)
{
    clear_neural_net();
    count_neural_net();
    recal_alpha();
    if (err() > MAX_ERR)
        adj_weight();
}

double NModel::err(void)
{
    double e = 0;
    for (int exp = 0; exp <= output_size; exp++)
    {
        e += abs(expected[exp] - neurons[inner_count + 1][exp]);
    }
    return e / 2;
}

void NModel::_res(int& size, double* arr)
{
    arr = neurons[inner_count + 1];
    size = output_size;
}

size_t NModel::apply(void)
{
    clear_neural_net();
    count_neural_net();
    int max_i = 0;
    double max_v = neurons[inner_count + 1][0];
    for (int i = 1; i < output_size; i++)
    {
        if (neurons[inner_count + 1][i] > max_v) {
            max_i = i;
            max_v = neurons[inner_count + 1][i];
        }
    }
    return max_i;
}
