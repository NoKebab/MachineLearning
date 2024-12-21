#ifndef MATRIX2D_HPP
#define MATRIX2D_HPP

#include <memory>
#include <vector>
#include <string>

//template<size_t _NUM_ROWS, size_t _NUM_COLS>
class Matrix2D
{
private:
    size_t rows;
    size_t cols;
    size_t length;
    //std::shared_ptr<double[]> elements;
    std::vector<double> elements;
    std::shared_ptr<Matrix2D> matrixTranspose;

    bool matrixSizesEqual(const Matrix2D &rhs) const;
public:
    struct Matrix2DShape
    {
        size_t rows;
        size_t cols;
    };

    Matrix2D(const size_t rows, const size_t cols);
    Matrix2D(std::vector<double> &vector);
    Matrix2D(std::vector<std::vector<double>> &vector);
    // create matrix with same shape
    Matrix2D(const Matrix2D &matrix);
    Matrix2D();
    ~Matrix2D();

    size_t getIndex(const size_t row, const size_t col) const;
    double getElement(const size_t row, const size_t col) const;
    void setElement(const size_t row, const size_t col, const double element);

    Matrix2D& add(const Matrix2D &rhs);
    // assuming rhs is a m x 1 matrix and this is a m x n matrix
    // add rhs to every column of this
    Matrix2D& addColumnWise(const Matrix2D &rhs);
    Matrix2D& sub(const Matrix2D &rhs);
    // element wise multiplication
    Matrix2D& mul(const Matrix2D &rhs);
    Matrix2D& div(const Matrix2D &rhs);
    Matrix2D& scale(const double scalar);

    // sum the elements row wise 0 or col wise 1
    Matrix2D& sum(const int axis);

    static void add(const Matrix2D &matrixA, const Matrix2D &matrixB, Matrix2D &matrixC);
    static void sub(const Matrix2D &matrixA, const Matrix2D &matrixB, Matrix2D &matrixC);
    static void mul(const Matrix2D &matrixA, const Matrix2D &matrixB, Matrix2D &matrixC);
    static void div(const Matrix2D &matrixA, const Matrix2D &matrixB, Matrix2D &matrixC);

    // matrix multipication
    static void matrixMultiply(const Matrix2D &matrixA, const Matrix2D &matrixB, Matrix2D &matrixC);
    // matrix multipication
    void matrixMultiply(const Matrix2D &rhs, Matrix2D &outputMatrix);

    //std::shared_ptr<double[]> getElements() { return elements; };
    std::vector<double>& getElements() { return elements; };
    size_t getLength() const { return length; };
    size_t getRows()   const { return rows; };
    size_t getCols()   const { return cols; };

    // apply an element wise function to the matrix
    Matrix2D& map(double (*mappingFunction)(const double));

    std::shared_ptr<Matrix2D> getTranspose();

    void show() const;
    void serialize(const std::string& filepath) const;
    std::string getStringRepresentation() const;

    Matrix2D& operator=(const Matrix2D& matrix);
    //Matrix2D operator+(const Matrix2D& matrix) const;
    //Matrix2D operator-(const Matrix2D& matrix) const;
    //Matrix2D operator*(const Matrix2D& matrix) const;
    //Matrix2D operator/(const Matrix2D& matrix) const;
    //Matrix2D& operator+=(const Matrix2D& matrix);
    //Matrix2D& operator-=(const Matrix2D& matrix);
    //Matrix2D& operator*=(const Matrix2D& matrix);
    //Matrix2D& operator/=(const Matrix2D& matrix);

    //Matrix2D& operator*(const double scalar);

    // get diagonal matrix values
    std::vector<double> diagonal() const;

    std::vector<double> getRow(const size_t rowIndex) const;
    std::vector<double> getCol(const size_t colIndex) const;
};

#endif // MATRIX2D_HPP