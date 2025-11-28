#pragma once
#include <vector>

class TimeSeriesData {
public:
    std::vector<double> timeGrid;
    std::vector<double> path;      
    double dt;                    
    
    TimeSeriesData() : dt(1.0/252.0) {}
    
    explicit TimeSeriesData(size_t reserveSize) : dt(1.0/252.0) {
        timeGrid.reserve(reserveSize);
        path.reserve(reserveSize);
    }
    
    TimeSeriesData(const std::vector<double>& times, const std::vector<double>& vals)
        : timeGrid(times), path(vals) {
        if (times.size() > 1) {
            dt = times[1] - times[0];
        } else {
            dt = 1.0/252.0;
        }
    }
    
    void push_back(double t, double value) {
        timeGrid.push_back(t);
        path.push_back(value);
    }
    
    [[nodiscard]] size_t size() const noexcept { 
        return path.size(); 
    }
    
    void clear() noexcept {
        timeGrid.clear();
        path.clear();
    }
};