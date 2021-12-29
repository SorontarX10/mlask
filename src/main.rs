use std::ops::Add;

#[derive(Debug)]
struct Mlarray {
    array: Vec<Vec<f64>>,
    n_dim: u32,
    shape: Vec<usize>,
    size: usize
}
impl Mlarray {
    fn new(array: Vec<Vec<f64>>) -> Mlarray {
        let n_dim: u32 = if array.len() == 1 { 1 } else { 2 };
        let shape = vec![array.len(), array[0].len()];
        let size = shape[0] * shape[1];
        Mlarray {array, n_dim, shape, size}
    }

    fn dot(&self, other: &Mlarray) -> Mlarray {
        println!("Shape of a: {:?}, shape of b: {:?}", self.shape, other.shape);
        if self.shape[1] != other.shape[0] {
            panic!("Shapes {:?} and {:?} mismatch for dot product", self.shape, other.shape);
        }

        let mut output_array: Vec<Vec<f64>> = Vec::new();
        for row_a in self.array.iter() {
            let mut inside_array = Vec::new();
            for i in 0..other.shape[1] {
                let mut row_result = 0.;
                for (row_b_index, col_a) in row_a.iter().enumerate() {
                    row_result += col_a * other.array[row_b_index][i];
                }
                inside_array.push(row_result);
            }
            output_array.push(inside_array);
        }
        let output_mlarray = Mlarray::new(output_array);
        output_mlarray
    }

    fn transpose(&mut self) -> Mlarray {
        // let mut outside_array: Vec<Vec<f64>> = Vec::new();
        let rows_number = self.shape[0];
        let columns_number = self.shape[1];

        let mut new_array: Vec<Vec<f64>> = Vec::new();
        let mut i = 0;
        while i < columns_number as i32 {
            let mut new_row: Vec<f64> = Vec::new();
            let mut j = 0;
            while j < rows_number as i32 {
                new_row.push(self.array[j as usize][i as usize]);
                j += 1;
            }
            new_array.push(new_row);
            i += 1;
        }
        Mlarray::new(new_array)
    }
}
impl Add for Mlarray {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut output_vector: Vec<Vec<f64>> = Vec::new();
        if self.shape == other.shape {
            for (a_row, b_row) in self.array
                .iter()
                .zip(other.array.iter()) {
                let mut inside_vector: Vec<f64> = Vec::new();
                for (a_val, b_val) in a_row
                    .iter()
                    .zip(b_row.iter()) {
                    inside_vector.push(a_val + b_val);
                }
                output_vector.push(inside_vector);
            }
            return Mlarray::new(output_vector)
        }
        if self.shape[1] == other.shape[1] {
            for a_row in self.array {
                let mut inside_vector: Vec<f64> = Vec::new();
                for (a_val, b_val) in a_row
                    .iter()
                    .zip(other.array[0].iter()) {
                    inside_vector.push(a_val + b_val);
                }
                output_vector.push(inside_vector);
            }
            return Mlarray::new(output_vector)
        }
        panic!("Shapes {:?} and {:?} mismatch for addition", self.shape, other.shape);
    }
}

fn main() {
    let inputs = vec![
        vec![1., 2., 3., 2.5],
        vec![2., 5., -1., 2.],
        vec![-1.5, 2.7, 3.3, -0.8]
        ];
    let weights = vec![
        vec![0.2, 0.8, -0.5, 1.],
        vec![0.5, -0.91, 0.26, -0.5],
        vec![-0.26, -0.27, 0.17, 0.87]
    ];
    let biases = vec![vec![2., 3., 0.5]];

    // let output;

    let inputs_arr = Mlarray::new(inputs);

    let mut weights_arr = Mlarray::new(weights);
    weights_arr = weights_arr.transpose();

    let biases_arr = Mlarray::new(biases);

    let dot_result = inputs_arr.dot(&weights_arr);
    println!("result: {:?}", dot_result);

    let result = dot_result + biases_arr;
    println!("Addition result: {:?}", result);
}
