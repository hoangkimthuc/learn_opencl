#include <iostream>
#include <vector>

using namespace std;

// Function to compute convolution
vector<vector<int>> convolution(const vector<vector<int>>& image, const vector<vector<int>>& filter) {
    int imageRows = image.size();
    int imageCols = image[0].size();
    int filterRows = filter.size();
    int filterCols = filter[0].size();
    int outputRows = imageRows - filterRows + 1;
    int outputCols = imageCols - filterCols + 1;

    vector<vector<int>> output(outputRows, vector<int>(outputCols, 0));

    for (int i = 0; i < outputRows; i++) {
        for (int j = 0; j < outputCols; j++) {
            int sum = 0;
            for (int k = 0; k < filterRows; k++) {
                for (int l = 0; l < filterCols; l++) {
                    sum += image[i + k][j + l] * filter[k][l];
                }
            }
            output[i][j] = sum;
        }
    }

    return output;
}

int main() {
    // Define a 4x4 image and a 2x2 filter
    vector<vector<int>> image = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    vector<vector<int>> filter = {
        {1, 1},
        {1, -1}
    };

    // Compute the convolution
    vector<vector<int>> result = convolution(image, filter);

    // Print the result
    for (const vector<int>& row : result) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}
