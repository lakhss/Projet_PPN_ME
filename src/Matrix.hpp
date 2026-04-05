#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>

class Matrix {
private:
    std::vector<double> data_;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;

public:
    Matrix() = default;

    Matrix(std::size_t rows, std::size_t cols)
        : data_(rows * cols), rows_(rows), cols_(cols) {}

    void resize(std::size_t rows, std::size_t cols) {
        rows_ = rows;
        cols_ = cols;
        data_.assign(rows * cols, 0.0);
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    bool empty() const { return rows_ == 0 || cols_ == 0; }

    double& operator()(std::size_t i, std::size_t j) {
        return data_[i * cols_ + j];
    }

    const double& operator()(std::size_t i, std::size_t j) const {
        return data_[i * cols_ + j];
    }

    const double* row_data(std::size_t i) const {
        return &data_[i * cols_];
    }

    double* row_data(std::size_t i) {
        return &data_[i * cols_];
    }

    std::vector<double> row(std::size_t i) const {
        return std::vector<double>(data_.begin() + i * cols_,
                                   data_.begin() + (i + 1) * cols_);
    }

    const std::vector<double>& raw() const {
        return data_;
    }
};