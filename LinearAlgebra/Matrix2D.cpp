#include "Matrix2D.hpp"
#include <iostream>
#include <cstring>

Matrix2D::Matrix2D()
    : rows(0)
    , cols(0)
    , length(0)
    , elements(length)
    , matrixTranspose(0)
{
}


Matrix2D::Matrix2D(const size_t rows, const size_t cols)
    : rows(rows)
    , cols(cols)
    , length(rows * cols)
    , elements(length)
    , matrixTranspose(0)
{
    memset(elements.data(), 0x00, sizeof(double) * length);
}


Matrix2D::Matrix2D(std::vector<double> &vector)
    : rows(vector.size())
    , cols(1)
    , length(rows * cols)
    //, elements(new double[length], std::default_delete<double[]>())
    , elements(length)
    , matrixTranspose(0)
{
    memcpy(elements.data(), vector.data(), sizeof(double) * length);
}


Matrix2D::Matrix2D(std::vector<std::vector<double>> &vector)
    : rows(vector.size())
    , cols(vector.front().size())
    , length(rows * cols)
    //, elements(new double[length], std::default_delete<double[]>())
    , elements(length)
    , matrixTranspose(0)
{
    for (size_t row = 0; row < rows; ++row)
    {
        for (size_t col = 0; col < cols; ++col)
        {
            setElement(row, col, vector[row][col]);
        }
    }
}


Matrix2D::Matrix2D(const Matrix2D &matrix)
    : rows(matrix.getRows())
    , cols(matrix.getCols())
    , length(matrix.getLength())
    //, elements(new double[length], std::default_delete<double[]>())
    , elements(length)
    , matrixTranspose(0)
{
    //memset(elements.data(), 0x00, sizeof(double) * length);
    elements = matrix.elements;
}

Matrix2D::~Matrix2D()
{
}

bool Matrix2D::matrixSizesEqual(const Matrix2D &rhs) const
{
    return (this->rows == rhs.rows) && (this->cols == rhs.cols);
}


size_t Matrix2D::getIndex(const size_t row, const size_t col) const
{
    return (row * cols) + col; 
}


double Matrix2D::getElement(const size_t row, const size_t col) const
{
    const size_t index = getIndex(row, col); 
    // prevent illegal memory access
    if (index >= length)
    {
        return 0;
    }
    return elements[index];
}


void Matrix2D::setElement(const size_t row, const size_t col, const double element)
{
    const size_t index = getIndex(row, col); 
    // prevent illegal memory access
    if (index < length)
    {
        elements[index] = element;
    }
}


Matrix2D& Matrix2D::add(const Matrix2D &rhs)
{
    if (matrixSizesEqual(rhs)) 
    {
        for (size_t i = 0; i < length; ++i)
        {
            elements[i] += rhs.elements[i];
        } 
    }
    else
    {
        std::cout << "Matrix2D::add() " << rows << "x" << cols << " != " << rhs.rows << "x" << rhs.cols << std::endl;
    }
    return *this;
}


Matrix2D& Matrix2D::sub(const Matrix2D &rhs)
{
    if (matrixSizesEqual(rhs)) 
    {
        for (size_t i = 0; i < length; ++i)
        {
            elements[i] -= rhs.elements[i];
        } 
    }
    else
    {
        std::cout << "Matrix2D::sub() " << rows << "x" << cols << " != " << rhs.rows << "x" << rhs.cols << std::endl;
    }
    return *this;
}


Matrix2D& Matrix2D::mul(const Matrix2D &rhs)
{
    if (matrixSizesEqual(rhs)) 
    {
        for (size_t i = 0; i < length; ++i)
        {
            elements[i] *= rhs.elements[i];
        } 
    }
    else
    {
        std::cout << "Matrix2D::mul() " << rows << "x" << cols << " != " << rhs.rows << "x" << rhs.cols << std::endl;
    }
    return *this;
}


Matrix2D& Matrix2D::div(const Matrix2D &rhs)
{
    if (matrixSizesEqual(rhs)) 
    {
        for (size_t i = 0; i < length; ++i)
        {
            // prevent divide by zero
            if (rhs.elements[i] != 0)
            {
                elements[i] /= rhs.elements[i];
            }
            else
            {
                elements[i] = 0;
            }
        } 
    }
    else
    {
        std::cout << "Matrix2D::div() " << rows << "x" << cols << " != " << rhs.rows << "x" << rhs.cols << std::endl;
    }
    return *this;
}


Matrix2D& Matrix2D::scale(const double scalar)
{
    for (size_t i = 0; i < length; ++i)
    {
        elements[i] *= scalar;
    }
    return *this;
}


void Matrix2D::add(const Matrix2D &matrixA, const Matrix2D &matrixB, Matrix2D &matrixC)
{
    if (matrixA.matrixSizesEqual(matrixB) && matrixB.matrixSizesEqual(matrixC))
    {
        for (size_t i = 0; i < matrixA.getLength(); ++i)
        {
            matrixC.elements[i] = matrixA.elements[i] + matrixB.elements[i];
        }
    }
}


void Matrix2D::sub(const Matrix2D &matrixA, const Matrix2D &matrixB, Matrix2D &matrixC)
{
    if (matrixA.matrixSizesEqual(matrixB) && matrixB.matrixSizesEqual(matrixC))
    {
        for (size_t i = 0; i < matrixA.getLength(); ++i)
        {
            matrixC.elements[i] = matrixA.elements[i] - matrixB.elements[i];
        }
    }
}


void Matrix2D::mul(const Matrix2D &matrixA, const Matrix2D &matrixB, Matrix2D &matrixC)
{
    if (matrixA.matrixSizesEqual(matrixB) && matrixB.matrixSizesEqual(matrixC))
    {
        for (size_t i = 0; i < matrixA.getLength(); ++i)
        {
            matrixC.elements[i] = matrixA.elements[i] * matrixB.elements[i];
        }
    }
}

/**
  * divide a by b and store in c
  */

void Matrix2D::div(const Matrix2D &matrixA, const Matrix2D &matrixB, Matrix2D &matrixC)
{
    if (matrixA.matrixSizesEqual(matrixB) && matrixB.matrixSizesEqual(matrixC))
    {
        for (size_t i = 0; i < matrixA.getLength(); ++i)
        {
            if (matrixB.elements[i] == 0)
            {
                matrixC.elements[i] = 0;
            }
            else
            {
                matrixC.elements[i] = matrixA.elements[i] / matrixB.elements[i];
            }
        }
    }
}

//Matrix2D& Matrix2D::matrix_multiply(const Matrix2D &rhs)


// C = AB for an n × m matrix A and an m × p matrix B, then C is an n × p matrix
//    Input: matrices A and B
//    Let C be a new matrix of the appropriate size
//    For i from 1 to n:
//        For j from 1 to p:
//            Let sum = 0
//            For k from 1 to m:
//                Set sum ← sum + Aik × Bkj
//            Set Cij ← sum
//    Return C
//

void Matrix2D::matrixMultiply(const Matrix2D &matrixA, const Matrix2D &matrixB, Matrix2D &matrixC)
{
    if (!((matrixA.cols == matrixB.rows) 
       && (matrixC.rows == matrixA.rows) 
       && (matrixC.cols == matrixB.cols))) 
    {
        printf("static Matrix2D::matrix_multiply() size mismatch A %zux%zu, B %zux%zu, C %zux%zu\n", matrixA.rows, matrixA.cols, matrixB.rows, matrixB.cols, matrixC.rows, matrixC.cols);
        return;
    }

    for (size_t i = 0; i < matrixA.rows; ++i)
    {
        for (size_t j = 0; j < matrixB.cols; ++j)
        {
            double sum = 0;
            for (size_t k = 0; k < matrixB.rows; ++k)
            {
                sum += matrixA.getElement(i, k) * matrixB.getElement(k, j);
            }        
            matrixC.setElement(i, j, sum);
        }        
    }        
}


void Matrix2D::matrixMultiply(const Matrix2D &rhs, Matrix2D &outputMatrix)
{
    if (!((cols == rhs.rows) 
       && (outputMatrix.rows == rows) 
       && (outputMatrix.cols == rhs.cols))) 
    {
        printf("Matrix2D::matrix_multiply() size mismatch A %zux%zu, B %zux%zu, C %zux%zu\n", rows, cols, rhs.rows, rhs.cols, outputMatrix.rows, outputMatrix.cols);
        return;
    }
    
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < rhs.cols; ++j)
        {
            double sum = 0;
            for (size_t k = 0; k < rhs.rows; ++k)
            {
                sum += getElement(i, k) * rhs.getElement(k, j);
            }        
            outputMatrix.setElement(i, j, sum);
        }        
    }        
}

// apply an element wise function to the matrix

Matrix2D& Matrix2D::map(double (*mappingFunction)(const double))
{
    for (size_t i = 0; i < length; ++i)
    {
        elements[i] = mappingFunction(elements[i]);
    }
    return *this;
}


void Matrix2D::show()
{
    std::cout << rows << "x" << cols << std::endl;
    for (size_t row = 0; row < rows; ++row)
    {
        for (size_t col = 0; col < cols; ++col)
        {
            if (col == (cols - 1))
            {
                printf("%.6f\n", getElement(row, col));
            }
            else
            {
                printf("%.6f,", getElement(row, col));
            }
        }
    }
}


std::string Matrix2D::getStringRepresentation()
{
    std::string s = "";
    //std::cout << rows << "x" << cols << std::endl;
    s += std::to_string(rows);
    s += "x";
    s += std::to_string(cols);
    s += "\n";
    for (size_t row = 0; row < rows; ++row)
    {
        for (size_t col = 0; col < cols; ++col)
        {
            s += std::to_string(getElement(row, col));
            if (col != cols - 1)
            {
                s += ",";
            }
            //printf("%.6f,", getElement(row, col));
            //std::cout << getElement(row, col) << ",";
        }
        s += "\n";
        //std::cout << std::endl;
    }
    return s;
}


Matrix2D& Matrix2D::operator=(const Matrix2D& matrix)
{
    rows = matrix.rows;
    cols = matrix.cols;
    length = matrix.length;

    elements = matrix.elements;
    //elements.clear();
    //for (double element : matrix.elements)
    //{
    //    elements.push_back(element);
    //}
    //elements = std::make_shared<double[]>();
    //elements.reset(new double[length], std::default_delete<double[]>());
    //elements(new double[length], std::default_delete<double[]>());
    //memcpy(elements.get(), matrix.elements.get(), sizeof(double) * length);
    return *this;
}

//
//Matrix2D Matrix2D::operator+(const Matrix2D& matrix) const
//{
//    Matrix2D outputMatrix(matrix);
//    if (matrixSizesEqual(matrix)) 
//    {
//        for (size_t i = 0; i < length; ++i)
//        {
//            outputMatrix.elements[i] = elements[i] + matrix.elements[i];
//        } 
//    }
//    return outputMatrix;
//}
//
//
//Matrix2D Matrix2D::operator-(const Matrix2D& matrix) const
//{
//    Matrix2D outputMatrix(matrix);
//    if (matrixSizesEqual(matrix)) 
//    {
//        for (size_t i = 0; i < length; ++i)
//        {
//            outputMatrix.elements[i] = elements[i] - matrix.elements[i];
//        } 
//    }
//    return outputMatrix;
//}
//
//
//Matrix2D Matrix2D::operator*(const Matrix2D& matrix) const
//{
//    Matrix2D outputMatrix(matrix);
//    if (matrixSizesEqual(matrix)) 
//    {
//        for (size_t i = 0; i < length; ++i)
//        {
//            outputMatrix.elements[i] = elements[i] * matrix.elements[i];
//        } 
//    }
//    return outputMatrix;
//}
//
//
//Matrix2D Matrix2D::operator/(const Matrix2D& matrix) const
//{
//    Matrix2D outputMatrix(matrix);
//    if (matrixSizesEqual(matrix)) 
//    {
//        for (size_t i = 0; i < length; ++i)
//        {
//            if (matrix.elements[i] != 0)
//            {
//                outputMatrix.elements[i] = elements[i] / matrix.elements[i];
//            }
//            else
//            {
//                outputMatrix.elements[i] = 0;
//            }
//        } 
//    }
//    return outputMatrix;
//}
//
//
//Matrix2D& Matrix2D::operator+=(const Matrix2D &matrix) 
//{
//    return add(matrix);
//}
//
//
//Matrix2D& Matrix2D::operator-=(const Matrix2D &matrix) 
//{
//    return sub(matrix);
//}
//
//
//Matrix2D& Matrix2D::operator*=(const Matrix2D &matrix) 
//{
//    return mul(matrix);
//}
//
//
//Matrix2D& Matrix2D::operator/=(const Matrix2D &matrix) 
//{
//    return div(matrix);
//}


std::shared_ptr<Matrix2D> Matrix2D::getTranspose() 
{
    if (0 == matrixTranspose)
    {
		matrixTranspose = std::make_shared<Matrix2D>(cols, rows);
    }
    matrixTranspose->elements = elements;
    return matrixTranspose;
}


//Matrix2D& Matrix2D::operator*(const double scalar)
//{
//    for (size_t i = 0; i < length; ++i)
//    {
//        elements[i] *= scalar;
//    }
//    return *this;
//}

std::vector<double> Matrix2D::diagonal() const
{
    std::vector<double> diagonalElements;
    if (rows == cols)
    {
        for (size_t i = 0; i < rows; ++i)
        {
            diagonalElements.push_back(getElement(i, i));
        }
    }
    else
    {
        std::cout << "Matrix2D::diagonal() matrix not square " << rows << "x" << cols << std::endl;
    }
    return diagonalElements;
}

std::vector<double> &Matrix2D::getRow(const size_t rowIndex) const
{
    std::vector<double> row;
    if (rowIndex >= rows)
    {
        std::cout << "Matrix2D::getRow() row index " << rowIndex << " out of bounds for num rows " << rows << std::endl;
        return row;
    }
    for (size_t col = 0; col < cols; ++col)
    {
        row.push_back(getElement(rowIndex, col));
    }
    return row;
}

std::vector<double>& Matrix2D::getCol(const size_t colIndex) const
{
    std::vector<double> col;
    if (colIndex >= cols)
    {
        std::cout << "Matrix2D::getCol() col index " << colIndex << " out of bounds for num cols " << cols << std::endl;
        return col;
    }
    for (size_t row = 0; row < rows; ++row)
    {
        col.push_back(getElement(row, colIndex));
    }
    return col;
}