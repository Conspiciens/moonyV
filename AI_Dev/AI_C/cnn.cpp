#include <iostream> 

using namespace std; 

class NeuralNetwork {
    private: 

    public: 

};


int main() {
    int weights[4][4] = {
        {0, 1, 0, 1}, 
        {1, 0, 1, 0}, 
        {1, 0, 1, 1}, 
        {1, 1, 0, 0} 
    }; 

    int kernel[3][3] = {
        {0, -1, 0}, 
        {-1, 5, -1}, 
        {0, -1, 0},
    };



    /* Convolution of kernel and weights O(n^4) LOL */
    for (int y1 = 0; y1 < 2; y1++){
        for (int x1 = 0; x1 < 2; x1++){

            for (int x = 0; x < 3; x++){
                for (int y = 0; y < 3; y++){

                    weights[x+x1][y + y1] =  kernel[x][y] * weights[x+x1][y+y1]; 

                }
            }

        }
    }



}