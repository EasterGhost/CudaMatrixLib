#include <iostream>
#include <vector>
using namespace std;

template <typename Type>
vector<Type> operator + (const vector<Type>& a, const vector<Type>& b)
{
	if (a.size() != b.size())
		throw runtime_error("Invalid vector addition.");
	vector<Type> result(a.size());
	for (int i = 0; i < a.size(); i++)
		result[i] = a[i] + b[i];
	return result;
}

template <typename Type>
vector<Type> operator - (const vector<Type>& a, const vector<Type>& b)
{
	if (a.size() != b.size())
		throw runtime_error("Invalid vector subtraction.");
	vector<Type> result(a.size());
	for (int i = 0; i < a.size(); i++)
		result[i] = a[i] - b[i];
	return result;
}

template <typename Type>
vector<Type> operator * (const vector<Type>& a, const vector<Type>& b)
{
	if (a.size() != b.size())
		throw runtime_error("Invalid vector multiplication.");
	vector<Type> result(a.size());
	for (int i = 0; i < a.size(); i++)
		result[i] = a[i] * b[i];
	return result;
}

template <typename Type>
vector<Type> operator / (const vector<Type>& a, const vector<Type>& b)
{
	if (a.size() != b.size())
		throw runtime_error("Invalid vector division.");
	vector<Type> result(a.size());
	for (int i = 0; i < a.size(); i++)
		result[i] = a[i] / b[i];
	return result;
}
