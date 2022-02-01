#include <iostream>
#include <algorithm>
#include <string>
#include <random>
#include <time.h>
#include <math.h>
#include <vector>https://github.com/kipflip1125/AI
#include "mpi.h"

#define CSV_IO_NO_THREAD
#include "csv.h"

// Data structure to hold all the features for the algorithm
struct data
{
    double glucose, BP, SkinThickness, insulin, BMI, outcome, distance;
};

std::vector<data> merge(std::vector<data> light, std::vector<data> right);
std::vector<data> mergeSort(std::vector<data> myvec);
void print_vec(std::vector<data> vec);
void check(double *t_labels, double *test, int len);


// prints out the expected and predicted values of the test data to see how accurate the model is
void check(double expected[], double predicted[], int len)
{   
    double corr = 0;
    for(int i = 0; i < len; i++)
    {
        if(expected[i] == predicted[i])
        {
            corr++;
        }
        printf("#%d\tExpected label: %f\tPredicted label: %f\n", i, expected[i], predicted[i]);
    }
    printf("Accuracy is %f\nError rate is %f\n", double(corr / len), 1-double(corr/len));
}

// Initialize the mpi variables
void mpiInitialize(int *size, int *rank)
{
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, size);
}

// the distance formula and store it into the structure
void distance(data* train, data *test)
{
    double dist = abs(train->glucose - test->glucose) + abs(train->BMI - test->BMI) + abs(train->BP - test->BP) + abs(train->insulin - test->insulin) + abs(train->SkinThickness - test->SkinThickness);
    train->distance = dist;
}

void knn(data *test_data, std::vector<data> *train_data, double *test_predicted, int size, int rank)
{
    int end = train_data->size() - 1;
    int data_per_process = train_data->size()/size;
       
    // allocate the space for the 
    data tmp[data_per_process];

    // Create an MPI datatype for our structure to handle MPI calls
    MPI_Datatype dt_data;
    MPI_Type_contiguous(7, MPI_DOUBLE, &dt_data);
    MPI_Type_commit(&dt_data);

    // Send parts of the training data to multiple processes and send each process the data to classify
    MPI_Scatter(&train_data->at(0), data_per_process, dt_data, &tmp, data_per_process, dt_data, 0, MPI_COMM_WORLD);\

    // each process needs to calculate distance and store it into the data struct
    for(int i = 0; i < data_per_process; i++)
    {
        distance(&tmp[i], test_data);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Gathers all the parts from the processes into the root rank
    
    MPI_Gather(&tmp, data_per_process, dt_data, &train_data->at(0), data_per_process, dt_data, 0, MPI_COMM_WORLD);

    // count the classes from top k values
    // whatever is max test point is whatever class was most
    if(rank == 0)
    {
        *train_data = mergeSort(*train_data);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    // send 1 data to each process and get the sum back
    // since the outcome is either a 1 or 0 adding the outcomes and seeing if that value is less than k/2 would result in having more 0 outcomes
    // and if the value is greater then there would be more 1 neighbors 
    double sum;
    MPI_Reduce(&train_data->at(rank).outcome, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if(rank == 0)
    {
        if(sum < size/2)
        {
            *test_predicted = 0;
        }
        else
        {
            *test_predicted = 1;
        }
        // std::cout << "Result: " << test_predicted << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// Split the dataset into a training and testing set
void split(std::vector<data> arr, std::vector<data> *test, std::vector<data> *train, double percent)
{
    int div = arr.size() * percent;
    
    for(int  i = 0; i < arr.size(); i++)
    {
        if(i > div)
            test->push_back(arr[i]);
        else
            train->push_back(arr[i]);
    }
}

// merging the dataset from lowest to highest
std::vector<data> merge(std::vector<data> left, std::vector<data> right)
{
    int lidx = 0, ridx = 0;
    std::vector<data> result;
    while(lidx < left.size() && ridx < right.size())
    {
        if(left.at(lidx).distance < right.at(ridx).distance)
        {
            result.push_back(left[lidx++]);
        }
        else
        {
            result.push_back(right[ridx++]);

        }
    }
    
    while(lidx < left.size())
    {
        result.push_back(left[lidx++]);
    }
    
    while(ridx < right.size())
    {
        result.push_back(right[ridx++]);

    }
    return result;
}

// Divide the vector up and perform the sort
std::vector<data> mergeSort(std::vector<data> myvec)
{
    if(myvec.size() <= 1)
    {
        return myvec;
    }
    int len = myvec.size() / 2;
    std::vector<data> left(myvec.begin(), myvec.begin() + len);
    std::vector<data> right(myvec.begin() + len, myvec.end());
    return merge(mergeSort(left), mergeSort(right));
}

// Print out the vector of data structs
void print_vec(std::vector<data> vec)
{
    for(int i = 0; i < vec.size(); i++)
    {
        printf("%d glucose: %f, BP: %f, ST: %f, insulin: %f, BMI: %f, outcome: %f, dist: %f\n", i, vec[i].glucose, vec[i].BP, vec[i].SkinThickness, vec[i].insulin, vec[i].BMI, vec[i].outcome, vec[i].distance);
    }
}

int main(int argc, char* argv[])
{
    // Create the data dataer values
    std::vector<data> myvec, train, test;
    

    // seed the randomizer
    unsigned seed = time(NULL);

    // read the csv file and store the values in a vector/array
    io::CSVReader<6> in("diabetes.csv");
    in.read_header(io::ignore_extra_column, "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Outcome");
    
    double glucose, BP, SkinThickness, insulin, BMI, outcome;
    while(in.read_row(glucose, BP, SkinThickness, insulin, BMI, outcome))
    {
        data tmp;
        tmp.glucose = glucose;
        tmp.BP = BP;
        tmp.SkinThickness = SkinThickness;
        tmp.insulin = insulin;
        tmp.BMI = BMI;
        tmp.outcome = outcome;
        myvec.push_back(tmp);
    }
    
    // create mpi variables and call to initialize them
    int size, rank;
    mpiInitialize(&size, &rank);

    // Shuffle the data in case the dataset is ordered
    std::shuffle(myvec.begin(), myvec.end(), std::default_random_engine(seed));
    
    // Split the dataset to training and testing
    split(myvec, &test, &train, 0.8);

    // Get test actual labels into an array
    double test_actual[test.size()], test_predicted[test.size()];
    for(int i = 0; i < test.size(); i++)
    {
        test_actual[i] = test[i].outcome;
    }

    // Perform the knn algorithm for the training set
    for(int idx = 0; idx < test.size(); idx++)
    {
        knn(&test.at(idx), &train, &test_predicted[idx], size, rank);
    }
    
    if(rank == 0)
    {
        check(test_actual, test_predicted, test.size());
    }

    // End mpi
    MPI_Finalize();
}
