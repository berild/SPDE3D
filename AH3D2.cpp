#include <cmath>
#include <vector> 

class AH
{
    public:
        AH(int numX, int numY, int numZ, const double (*H)[6][3][3],double hx,double hy, double hz); //constructor
        int* Row();
        int* Col();
        double* Val();
        int* row;
        int* col;
        double* val;
        ~AH(); //destructor
};
// constuctor 
AH::AH(int numX, int numY, int numZ, const double (*H)[6][3][3],double hx,double hy, double hz)
{
    int idx = 0;
    row = new int [numX*numY*numZ*19];
    col = new int [numX*numY*numZ*19];
    val = new double [numX*numY*numZ*19];
    int k = 0;
    for(int j = 0; j < numY; j++){
        for(int i = 0; i < numX; i++){
            for (int m = 0; m < numZ; m++){
                int i_n = i - 1;
                int i_p = i + 1;
                int j_n = j - 1;
                int j_p = j + 1;
                int m_p = m + 1;
                int m_n = m - 1;
                if ( i == 0 ) {
                    i_n = i;
                }else if ( i == (numX - 1) ){
                    i_p = i;
                }
                if ( j == 0 ){
                    j_n = j;
                }else if ( j == (numY - 1) ){
                    j_p = j;
                }
                if ( m == 0 ){
                    m_n = m;
                }else if ( m == (numZ - 1) ){
                    m_p = m;
                }
                k = j*numX*numZ + i*numZ + m;
                for (int is=idx; is<(idx+19);++is){
                    row[is] = k;
                }
                
                col[idx] = k;
                //z
                col[idx + 1] = int(j*numX*numZ + i*numZ + m_p);
                col[idx + 2] = int(j*numX*numZ + i*numZ + m_n);
                //x
                col[idx + 3] = int(j*numX*numZ + i_p*numZ + m);
                col[idx + 4] = int(j*numX*numZ + i_n*numZ + m);
                //y
                col[idx + 5] = int(j_p*numX*numZ + i*numZ + m);
                col[idx + 6] = int(j_n*numX*numZ + i*numZ + m);
                //z and x
                col[idx + 7] = int(j*numX*numZ + i_p*numZ + m_p);
                col[idx + 8] = int(j*numX*numZ + i_n*numZ + m_n);
                col[idx + 9] = int(j*numX*numZ + i_n*numZ + m_p);
                col[idx + 10] = int(j*numX*numZ + i_p*numZ + m_n);
                //z and y
                col[idx + 11] = int(j_p*numX*numZ + i*numZ + m_p);
                col[idx + 12] = int(j_n*numX*numZ + i*numZ + m_n);
                col[idx + 13] = int(j_n*numX*numZ + i*numZ + m_p);
                col[idx + 14] = int(j_p*numX*numZ + i*numZ + m_n);
                //x and y
                col[idx + 15] = int(j_p*numX*numZ + i_p*numZ + m);
                col[idx + 16] = int(j_n*numX*numZ + i_n*numZ + m);
                col[idx + 17] = int(j_n*numX*numZ + i_p*numZ + m);
                col[idx + 18] = int(j_p*numX*numZ + i_n*numZ + m);
                
                //H indicies i-1/2, i+1/2, j-1/2, j+1/2, k-1/2, k+1/2

                val[idx] = - hy*hz/hx*(H[k][1][0][0] + H[k][0][0][0]) - hx*hz/hy*(H[k][3][1][1] + H[k][2][1][1]) - hx*hy/hz*(H[k][5][2][2] + H[k][4][2][2]);
                //z
                val[idx + 1] = hx*hy/hz*H[k][5][2][2] + hy/4.0*(H[k][1][2][0] - H[k][0][2][0]) + hx/4.0*(H[k][3][2][1] - H[k][2][2][1]);
                val[idx + 2] = hx*hy/hz*H[k][4][2][2] - hy/4.0*(H[k][1][2][0] - H[k][0][2][0]) - hx/4.0*(H[k][3][2][1] - H[k][2][2][1]);
                //x
                val[idx + 3] = hy*hz/hx*H[k][1][0][0] + hy/4.0*(H[k][5][0][1] - H[k][4][0][1]) + hz/4.0*(H[k][3][0][2] - H[k][2][0][2]);
                val[idx + 4] = hy*hz/hx*H[k][0][0][0] - hy/4.0*(H[k][5][0][1] - H[k][4][0][1]) - hz/4.0*(H[k][3][0][2] - H[k][2][0][2]);
                //y
                val[idx + 5] = hx*hz/hy*H[k][3][1][1] + hx/4.0*(H[k][5][1][2] - H[k][4][1][2]) + hz/4.0*(H[k][1][1][0] - H[k][0][1][0]);
                val[idx + 6] = hx*hz/hy*H[k][2][1][1] - hx/4.0*(H[k][5][1][2] - H[k][4][1][2]) - hz/4.0*(H[k][1][1][0] - H[k][0][1][0]);
                //z and x
                val[idx + 7] = hy/4.0*(H[k][1][2][0] + H[k][5][0][2]);
                val[idx + 8] = hy/4.0*(H[k][0][2][0] + H[k][4][0][2]);
                val[idx + 9] = -hy/4.0*(H[k][0][2][0] + H[k][5][0][2]);
                val[idx + 10] = -hy/4.0*(H[k][1][2][0] + H[k][4][0][2]);
                //z and y
                val[idx + 11] = hx/4.0*(H[k][3][2][1] + H[k][5][1][2]);
                val[idx + 12] = hx/4.0*(H[k][2][2][1] + H[k][4][1][2]);
                val[idx + 13] = -hx/4.0*(H[k][2][2][1] + H[k][5][1][2]);
                val[idx + 14] = -hx/4.0*(H[k][3][2][1] + H[k][4][1][2]);
                //x and y
                val[idx + 15] = hz/4.0*(H[k][1][1][0] + H[k][3][0][1]);
                val[idx + 16] = hz/4.0*(H[k][0][1][0] + H[k][2][0][1]);
                val[idx + 17] = -hz/4.0*(H[k][1][1][0] + H[k][2][0][1]);
                val[idx + 18] = -hz/4.0*(H[k][0][1][0] + H[k][3][0][1]);

                idx += 19;
            }
        }
    }
}

AH::~AH(){
    delete[] row;
    delete[] col;
    delete[] val;
}

// get row indices
int* AH::Row()
{
    return row;
}

// get row indices
int* AH::Col()
{
    return col;
}

// get row values
double* AH::Val()
{
    return val;
}
// Define C functions for the C++ class - as ctypes can only talk to C...
extern "C"
{
    AH* AH_new(int numX, int numY, int numZ, const double (*H)[6][3][3], double hx, double hy, double hz){
        return new AH(numX, numY, numZ, H, hx, hy, hz);}
    int* AH_Row(AH* ah) {return ah->Row();}
    int* AH_Col(AH* ah) {return ah->Col();}
    double* AH_Val(AH* ah) {return ah->Val();}
    void AH_delete(AH* ah)
    {
        delete ah;
    }
}

//g++ -c -fPIC AH3D2.cpp -o AH3D2.o
//g++ -shared -W1,libAH3D2.so -o libAH3D2.so AH3D2.o
//g++ -shared -o libAH3D2.so AH3D2.o