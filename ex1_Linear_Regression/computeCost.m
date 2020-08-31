function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

H = (theta' * X')'; % H = (multiply X and theta, in the proper order that the inner dimensions match); 
                    % X is size (m x n) and theta is size (n x 1), we arrange the order of operators so the result is size (m x 1).

training_error = H - y; % Compute the difference between the hypothesis and y - that's the error for each training example.

error_sqr = training_error .^ 2 ; % Compute the square of each of those error terms (using element-wise exponentiation)

S = sum(error_sqr) % compute the sum of the error_sqr vector

J = S / (2 * m)  % and scale the result (multiply) by 1/(2*m). That completed sum is the cost value J


% =========================================================================

end
