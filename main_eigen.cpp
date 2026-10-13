#include <iostream>
#include <Eigen/Dense>
//#include <unsupported/Eigen/MatrixFunctions>
//using namespace Eigen;

using matrix=Eigen::MatrixXi;

matrix cyclic_shift(int n);
matrix tensor_product(matrix a, matrix b);
matrix generate_x(int l, int m);
matrix generate_y(int l, int m);
matrix input_matrix(int l, int m);
matrix generate_xy(char c,int l, int m);
matrix matrix_power(const matrix& A, int p);

int main() {
    int l,m,n;
    matrix a,b;
    /*
    std::cout<<"Enter l:";
    std::cin>>l;
    std::cout<<"Enter m:";
    std::cin>>m;

    std::cout<<"Enter A (format is x 4 y 12 y 1):";
    a = input_matrix(l,m);
    std::cout<<"Enter B (format is x 4 y 12 y 1):";
    b = input_matrix(l,m);
    std::cout<<"\n"<<a<<"\n\n"<<b<<"\n\n";
    */
    l=6,m=7;
    n=2*l*m;
    a=matrix_power(generate_x(l,m),5)+matrix_power(generate_y(l,m),2)+matrix_power(generate_y(l,m),4);
    b=matrix_power(generate_x(l,m),1)+matrix_power(generate_x(l,m),3)+matrix_power(generate_y(l,m),5);
    //std::cout<<a<<"\n\n"<<b<<"\n\n";
    //a=a.array() - (2 * (a.array()/2));
    //b=b.array() - (2 * (b.array()/2));
    //std::cout<<"div 2"<<(a.array()/2)<<"\n\n";
    //std::cout<<"mul div 2"<<(2 * (a.array()/2))<<"\n\n";
    //std::cout<<"mul div 2 -"<<(a.array() - (2 * (a.array()/2)))<<"\n\n";
    a = a.unaryExpr([](int value) {
    return value % 2;});
    b = b.unaryExpr([](int value) {
    return value % 2;});
    //std::cout<<a<<"\n\n"<<b<<"\n\n";
    matrix hx(n/2,n);
    matrix hz(n/2,n);

    hx<<a,b;
    hz<<b.transpose(),a.transpose();

    //std::cout<<hx<<"\n\n"<<hz<<"\n\n";
    //std::cout<<hx*(hz.transpose())<<"\n\n";
}
matrix matrix_power(const matrix& A, int p)
{
    if (p < 0 || A.rows() != A.cols())
        throw std::invalid_argument(
            "Requires a square matrix and nonnegative exponent");

    matrix result = matrix::Identity(A.rows(), A.cols());

    for (int i = 0; i < p; ++i)
        result = result * A;

    return result;
}

matrix input_matrix(int l, int m)
{
    matrix a = Eigen::MatrixXi::Zero(l*m,l*m);
    for (int i = 0; i < 3; ++i)
    {
        char c;
        int x;
        std::cin>>c>>x;
        if (c=='x')
            a += matrix_power(generate_x(l,m),x);
        else if (c=='y')
            a += matrix_power(generate_y(l,m),x);
        else
            throw std::invalid_argument("Wrong input");
    }
    return a;
}

matrix cyclic_shift(int n)
{
    matrix a = Eigen::MatrixXi::Zero(n,n);
    for (int i = 0; i < n; ++i) {
        a(i, (i + 1) % n) = 1;
    }
    return a;
}

matrix tensor_product(matrix a, matrix b)
{
    int ax=a.rows(),ay=a.cols(),bx=b.rows(),by=b.cols();
    matrix c = Eigen::MatrixXi::Zero(ax*bx,ay*by);
    for (int i = 0; i < ax; ++i)
    {
        for (int j = 0; j < ay; ++j)
        {
            if (a(i,j)==1)
            {
                for (int k = 0; k < bx; ++k)
                {
                    for (int l = 0; l < by; ++l)
                    {
                        c(i*bx+k,j*by+l) = b(k,l);
                    }
                }
            }
        }
    }
    return c;
}

matrix generate_x(int l, int m)
{
    matrix a = cyclic_shift(l);
    matrix b = Eigen::MatrixXi::Identity(m,m);
    return tensor_product(a, b);
}


matrix generate_y(int l, int m)
{
    matrix a = Eigen::MatrixXi::Identity(l,l);
    matrix b = cyclic_shift(m);
    return tensor_product(a, b);
}

matrix generate_xy(char c, int l, int m)
{
    if (c=='x')
        return generate_x(l,m);
    else if (c=='y')
        return generate_y(l,m);
    throw std::invalid_argument("Wrong input");
}

matrix kernel_gf2(matrix H)
{
    int m = H.rows();
    int n = H.cols();
    int rank = 0;

    std::vector<int> pivots;
    std::vector<bool> is_pivot(n, false);

    // Row reduction over GF(2)
    for (int col = 0; col < n && rank < m; ++col) {
        int pivot = rank;

        while (pivot < m && H(pivot, col) == 0)
            ++pivot;

        if (pivot == m)
            continue;

        if (pivot != rank)
            H.row(rank).swap(H.row(pivot));

        for (int row = 0; row < m; ++row) {
            if (row != rank && H(row, col) == 1) {
                for (int j = col; j < n; ++j)
                    H(row, j) ^= H(rank, j); // Addition modulo 2
            }
        }

        pivots.push_back(col);
        is_pivot[col] = true;
        ++rank;
    }

    // One basis vector per free column
    matrix K = Eigen::MatrixXi::Zero(n, n - rank);
    int basis_col = 0;

    for (int col = 0; col < n; ++col) {
        if (is_pivot[col])
            continue;

        K(col, basis_col) = 1;

        for (int row = 0; row < rank; ++row)
            K(pivots[row], basis_col) = H(row, col);

        ++basis_col;
    }

    return K;
}