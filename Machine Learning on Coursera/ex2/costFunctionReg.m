function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
z=X*theta;
h=sigmoid(z);

theta0=theta(1);
theta(1)=0;

part3=lambda/(2*m)*(theta'*theta);


part1=-1*(y')*log(h);
part2=-1*((1.-y)')*log(1.-h);
J=(1/m)*(part1+part2)+part3;


gradinit=(1/m)*(X')*(h-y);

grad=gradinit+lambda/(m)*theta;
grad(1)=gradinit(1);


% =============================================================

end
