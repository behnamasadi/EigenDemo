#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>


#include <iostream>
#include <vector>
////////////////////////////C++ Functor////////////////////////////
/*** print the name of some types... ***/

template<typename type>
std::string name_of_type()
{
    return "other";
}

template<>
std::string name_of_type<int>()
{
    return "int";
}

template<>
std::string name_of_type<float>()
{
    return "float";
}

template<>
std::string name_of_type<double>()
{
    return "double";
}

template<typename scalar>
struct product_functor
{
    product_functor(scalar a, scalar b) : m_a(a), m_b(b)
    {
        std::cout << "Type: " << name_of_type<scalar>() << ". Computing the product of " << a << " and " << b << ".";
    }
    // the objective function a*b
    scalar f() const
    {
        return m_a * m_b;
    }

private:
    scalar m_a, m_b;
};

struct sum_of_ints_functor
{
    sum_of_ints_functor(int a, int b) : m_a(a), m_b(b)
    {
        std::cout << "Type: int. Computing the sum of the two ints " << a << " and " << b << ".";
    }

    int f() const
    {
        return m_a + m_b;
    }

    private:
    int m_a, m_b;
};

template<typename functor_type>
void call_and_print_return_value(const functor_type& functor_object)
{
    std::cout << " The result is: " << functor_object.f() << std::endl;
}

void functorExample()
{
    call_and_print_return_value(sum_of_ints_functor(3,5));
    call_and_print_return_value(product_functor<float>(0.2f,0.4f));
}


int main()
{
    functorExample();
}

