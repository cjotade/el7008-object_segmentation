#define _DEBUG

// Instruciones:
// Dependiendo de la versión de opencv, pueden cambiar los archivos .hpp a usar

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <numeric>
#include <algorithm>

using namespace std;

vector<vector<string> > parseCSV(string path_csv){

    std::ifstream data(path_csv.c_str());
    std::string line;
    vector<vector<string> > parsedCsv;
    while(std::getline(data,line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        vector<string> parsedRow;
        while(std::getline(lineStream,cell,' ')){
            parsedRow.push_back(cell);
            
        }
        parsedCsv.push_back(parsedRow);
    }
    
    return parsedCsv;
}

int main(void){
    string path_in = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/images_pgm/img_info_LR.txt";

    vector<vector<string> > parsedCsv = parseCSV(path_in);
    for ( const vector<string> &v : parsedCsv )
    {
        cout << "Inner Loop" << endl;

        for ( string x : v )
        {
            cout << x << ' ';
        }
        std::cout << std::endl;
    }

    cout << parsedCsv[0].size() << endl;
    cout << parsedCsv[0][0] << endl; //left
    cout << parsedCsv[0][5] << endl; //right
    return 0; 
}