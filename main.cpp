#include <bit>
#include <iostream>
#include <stdexcept>
#include <NTL/mat_GF2.h>
#include <chrono>
#include <iomanip>

using matrix = NTL::mat_GF2;

matrix cyclic_shift(long n);
matrix tensor_product(const matrix& a, const matrix& b);
matrix generate_x(long l, long m);
matrix generate_y(long l, long m);
matrix input_matrix(long l, long m);
matrix generate_xy(char c, long l, long m);
matrix matrix_power(const matrix& a, long p);
matrix concatenate_horizontal(const matrix& a, const matrix& b);
matrix kernel_gf2(const matrix& h);

//int find_d(const matrix& basis1, const matrix& basis2);
long find_d(const matrix& logical_basis,
            const matrix& stabilizer_basis);
int main()
{
    try {
        long l = 6;
        long m = 6;
        matrix a, b;

        /*
        std::cout << "Enter l: ";
        if (!(std::cin >> l))
            throw std::invalid_argument("Invalid l");
        std::cout << "Enter m: ";
        if (!(std::cin >> m))
            throw std::invalid_argument("Invalid m");

        std::cout << "Enter A (three terms, e.g. x 4 y 12 y 1): ";
        a = input_matrix(l, m);
        std::cout << "Enter B (three terms, e.g. x 4 y 12 y 1): ";
        b = input_matrix(l, m);
        */

        a =  NTL::power(generate_x(l, m), 3)
          + NTL::power(generate_y(l, m), 1)
          + NTL::power(generate_y(l, m), 2);

        b = NTL::power(generate_x(l, m), 1)
          + NTL::power(generate_x(l, m), 2)
          + NTL::power(generate_y(l, m), 3);

        const long n = 2 * l * m;

        matrix hx = concatenate_horizontal(a, b);
        matrix hz = concatenate_horizontal(NTL::transpose(b), NTL::transpose(a));

        matrix commutation = hx * NTL::transpose(hz);
        if (!NTL::IsZero(commutation))
            throw std::runtime_error("HX and HZ do not commute");

        matrix kx = kernel_gf2(hx);

        matrix basis;
        image(basis, hz);
        long rank = basis.NumRows();

        matrix basis_removed;
        basis_removed.SetDims(0,n);
        int basis_removed_num = 0;
        int i=0;
        while(i<kx.NumRows())
        {
            basis.SetDims(rank+1,n);
            basis[rank]=kx[i];
            matrix temp(basis);
            image(basis,temp);
            long new_rank = basis.NumRows();
            if (new_rank != rank)
            {
                basis_removed.SetDims(basis_removed_num+1,n);
                basis_removed[basis_removed_num++]=kx[i];
            }
            rank = new_rank;
            i++;
        }

        image(basis, hz);
        long d=find_d(basis_removed,basis);


        std::cout<<d;
        /*
        matrix hx_copy = hx;
        matrix hz_copy = hz;
        const long rank_x = NTL::gauss(hx_copy);
        const long rank_z = NTL::gauss(hz_copy);

        std::cout << "Physical qubits: " << n << '\n';
        std::cout << "HX shape: " << hx.NumRows() << " x " << hx.NumCols() << '\n';
        std::cout << "HZ shape: " << hz.NumRows() << " x " << hz.NumCols() << '\n';
        std::cout << "HX * HZ^T = 0 over GF(2): yes\n";
        std::cout << "Rank HX: " << rank_x << '\n';
        std::cout << "Rank HZ: " << rank_z << '\n';
        std::cout << "Logical qubits: " << n - rank_x - rank_z << '\n';
        std::cout << "Kernel HX dimension: " << kx.NumCols() << '\n';
        std::cout << "Kernel HZ dimension: " << kz.NumCols() << '\n';

        // NTL prints matrices as bracketed rows.
        std::cout << "A:\n" << a << "\nB:\n" << b << '\n';
        std::cout << "HX:\n" << hx << "\nHZ:\n" << hz << '\n';
        std::cout << "Kernel HX (basis in columns):\n" << kx << '\n';
        */


        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}

/*int find_d(const matrix& basis1, const matrix& basis2)
{
    size_t n,m,d=SIZE_MAX;
    n=basis1.NumRows(),m=basis2.NumRows();

    size_t total1=1ULL<<n,total2=1ULL<<n;
    NTL::Vec<NTL::GF2> cur_vec1;
    cur_vec1.SetLength(basis1.NumCols());

    for (size_t i=1;i<total1;++i)
    {
        NTL::Vec<NTL::GF2> cur_vec2;
        cur_vec2.SetLength(basis2.NumCols());
        for (size_t j=1;j<total2;++j)
        {
            size_t changed_elem=__builtin_ctzll(j);
            cur_vec2+=basis1[changed_elem];
            auto t=cur_vec1+cur_vec2;
            size_t dd=0;
            for (auto e :t)
                if (IsOne(e))dd++;
            d= std::min(d,dd);
        }
        size_t changed_elem=__builtin_ctzll(i);
        cur_vec1+=basis1[changed_elem];
    }
    NTL::Vec<NTL::GF2> cur_vec2;
    cur_vec2.SetLength(basis2.NumCols());
    for (size_t j=1;j<total2;++j)
    {
        size_t changed_elem=__builtin_ctzll(j);
        cur_vec2+=basis1[changed_elem];
        auto t=cur_vec1+cur_vec2;
        size_t dd=0;
        for (auto e :t)
            if (IsOne(e))dd++;
        d= std::min(d,dd);
    }
    return d;
}*/
long find_d(const matrix& logical_basis,
            const matrix& stabilizer_basis)
{
    const long k = logical_basis.NumRows();
    const long r = stabilizer_basis.NumRows();

    if (logical_basis.NumCols() != stabilizer_basis.NumCols())
        throw std::invalid_argument("Vector lengths differ");

    if (k == 0)
        throw std::invalid_argument("No nontrivial logical operators");

    if (k >= 64 || r >= 64)
        throw std::invalid_argument("Too many rows for 64-bit enumeration");

    const std::uint64_t total1 = std::uint64_t{1} << k;
    const std::uint64_t total2 = std::uint64_t{1} << r;

    long d = logical_basis.NumCols();

    auto last_update = std::chrono::steady_clock::now();

    auto show_progress = [&](std::uint64_t i, std::uint64_t j) {
        auto now = std::chrono::steady_clock::now();


        if (now - last_update < std::chrono::milliseconds(250))
            return;

        last_update = now;

        long double completed =
            static_cast<long double>(i - 1)
            + (static_cast<long double>(j) + 1) / total2;

        long double percent = 100.0L * completed / (total1 - 1);

        std::cerr << "\nProgress: "
                  << std::fixed << std::setprecision(1)
                  << percent << "%  Best weight: " << d
                  << "    " << std::flush;
    };

    NTL::vec_GF2 logical;
    logical.SetLength(logical_basis.NumCols()); // Initially zero

    for (std::uint64_t i = 1; i < total1; ++i) {
        logical += logical_basis[std::countr_zero(i)];

        NTL::vec_GF2 candidate = logical;

        d = std::min(d, NTL::weight(candidate));
        show_progress(i,0);
        for (std::uint64_t j = 1; j < total2; ++j) {
            candidate += stabilizer_basis[std::countr_zero(j)];
            d = std::min(d, NTL::weight(candidate));
            if (j % 65536  == 0)
            {
                show_progress(i, j);
            }

        }
    }

    return d;
}

matrix input_matrix(long l, long m)
{
    if (l <= 0 || m <= 0)
        throw std::invalid_argument("l and m must be positive");

    matrix a;
    a.SetDims(l * m, l * m); // New GF(2) entries are zero.

    for (int i = 0; i < 3; ++i) {
        char c;
        long p;
        if (!(std::cin >> c >> p))
            throw std::invalid_argument("Expected a term such as x 4");
        a += NTL::power(generate_xy(c, l, m), p);
    }

    return a;
}

matrix cyclic_shift(long n)
{
    if (n <= 0)
        throw std::invalid_argument("Shift size must be positive");

    matrix a;
    a.SetDims(n, n);
    for (long i = 0; i < n; ++i)
        a[i][(i + 1) % n] = 1;
    return a;
}

matrix tensor_product(const matrix& a, const matrix& b)
{
    const long ax = a.NumRows(), ay = a.NumCols();
    const long bx = b.NumRows(), by = b.NumCols();

    matrix c;
    c.SetDims(ax * bx, ay * by);

    for (long i = 0; i < ax; ++i) {
        for (long j = 0; j < ay; ++j) {
            if (NTL::IsOne(a[i][j])) {
                for (long k = 0; k < bx; ++k)
                    for (long t = 0; t < by; ++t)
                        c[i * bx + k][j * by + t] = b[k][t];
            }
        }
    }
    return c;
}

matrix generate_x(long l, long m)
{
    if (l <= 0 || m <= 0)
        throw std::invalid_argument("l and m must be positive");
    return tensor_product(cyclic_shift(l), NTL::ident_mat_GF2(m));
}

matrix generate_y(long l, long m)
{
    if (l <= 0 || m <= 0)
        throw std::invalid_argument("l and m must be positive");
    return tensor_product(NTL::ident_mat_GF2(l), cyclic_shift(m));
}

matrix generate_xy(char c, long l, long m)
{
    if (c == 'x') return generate_x(l, m);
    if (c == 'y') return generate_y(l, m);
    throw std::invalid_argument("Expected x or y");
}

matrix concatenate_horizontal(const matrix& a, const matrix& b)
{
    if (a.NumRows() != b.NumRows())
        throw std::invalid_argument("Matrices must have equal row counts");

    matrix c;
    c.SetDims(a.NumRows(), a.NumCols() + b.NumCols());

    for (long i = 0; i < c.NumRows(); ++i) {
        for (long j = 0; j < a.NumCols(); ++j)
            c[i][j] = a[i][j];
        for (long j = 0; j < b.NumCols(); ++j)
            c[i][a.NumCols() + j] = b[i][j];
    }
    return c;
}

matrix kernel_gf2(const matrix& h)
{
    // NTL returns row vectors v satisfying v * A = 0.
    // Set A = H^T
    matrix row_basis;
    NTL::kernel(row_basis, NTL::transpose(h));
    return row_basis;
}
