// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{Array, ArrayRef};
use arrow::datatypes::{
    ArrowNativeTypeOp, DataType, Decimal128Type, Decimal256Type, Decimal32Type,
    Decimal64Type, Int64Type,
};
use arrow::error::ArrowError;

use std::any::Any;
use std::mem::swap;
use std::ops::RemAssign;

use crate::utils::{calculate_binary_math, calculate_binary_math_decimal};
use datafusion_common::utils::take_function_args;
use datafusion_common::{exec_err, Result};
use datafusion_expr::type_coercion::is_decimal;
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_macros::user_doc;
use log::info;
use num_traits::{CheckedNeg, Signed};

#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns the greatest common divisor of `expression_x` and `expression_y`. Returns 0 if both inputs are zero.",
    syntax_example = "gcd(expression_x, expression_y)",
    sql_example = r#"```sql
> SELECT gcd(48, 18);
+------------+
| gcd(48,18) |
+------------+
| 6          |
+------------+
```"#,
    standard_argument(name = "expression_x", prefix = "First numeric"),
    standard_argument(name = "expression_y", prefix = "Second numeric")
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct GcdFunc {
    signature: Signature,
}

impl Default for GcdFunc {
    fn default() -> Self {
        Self::new()
    }
}

impl GcdFunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for GcdFunc {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "gcd"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [arg1, arg2] = take_function_args(self.name(), arg_types)?;

        fn coerced_type_impl(name: &str, data_type: &DataType) -> Result<DataType> {
            match data_type {
                // TODO: check if null is supported
                DataType::Null => Ok(DataType::Int64),
                d if d.is_integer() => Ok(DataType::Int64),
                d if is_decimal(d) => Ok(d.clone()),
                other => {
                    exec_err!("Unsupported data type {other:?} for {} function", name)
                }
            }
        }

        Ok(vec![
            coerced_type_impl(self.name(), arg1)?,
            coerced_type_impl(self.name(), arg2)?,
        ])
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let left = &args.args[0].to_array(args.number_rows)?;
        let right = &args.args[1];

        info!("invoke gcd with {left:?} and {right:?}");
        let out_type = left.data_type();

        let arr: ArrayRef = match (left.data_type(), right.data_type()) {
            (DataType::Int64, _) => calculate_binary_math::<
                Int64Type,
                Int64Type,
                Int64Type,
                _,
            >(&left, right, euclid_gcd_with_unsigned)?,
            (DataType::Decimal32(_, _), DataType::Decimal32(_, _)) => {
                calculate_binary_math_decimal::<
                    Decimal32Type,
                    Decimal32Type,
                    Decimal32Type,
                    _,
                >(&left, right, |a, b, _| euclid_gcd(a, b), out_type)?
            }
            (DataType::Decimal64(_, _), DataType::Decimal64(_, _)) => {
                calculate_binary_math_decimal::<
                    Decimal64Type,
                    Decimal64Type,
                    Decimal64Type,
                    _,
                >(&left, right, |a, b, _| euclid_gcd(a, b), out_type)?
            }
            (DataType::Decimal128(_, _), DataType::Decimal128(_, _)) => {
                calculate_binary_math_decimal::<
                    Decimal128Type,
                    Decimal128Type,
                    Decimal128Type,
                    _,
                >(&left, right, |a, b, _| euclid_gcd(a, b), out_type)?
            }
            (DataType::Decimal256(_, _), DataType::Decimal256(_, _)) => {
                calculate_binary_math_decimal::<
                    Decimal256Type,
                    Decimal256Type,
                    Decimal256Type,
                    _,
                >(&left, right, |a, b, _| euclid_gcd(a, b), out_type)?
            }
            (base_type, exp_type) => {
                return exec_err!(
                    "Unsupported data types for base {base_type:?} and exponent {exp_type:?} for function {}",
                    self.name()
                );
            }
        };
        Ok(ColumnarValue::Array(arr))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

/// Generic version gcd of two signed integers
/// Resorts to euclid_gcd_unsigned if arguments fit
fn euclid_gcd<T>(a: T, b: T) -> Result<T, ArrowError>
where
    T: ArrowNativeTypeOp + RemAssign + Signed + CheckedNeg,
{
    let a = if a.is_positive() {
        a
    } else {
        a.checked_neg()
            .ok_or_else(|| ArrowError::ComputeError("Signed integer overflow".into()))?
    };
    let b = if b.is_positive() {
        b
    } else {
        b.checked_neg()
            .ok_or_else(|| ArrowError::ComputeError("Signed integer overflow".into()))?
    };
    // Fall back to unsigned gcd
    euclid_gcd_unsigned(a, b)
}

/// Generic version of Euclidean algorithm to compute gcd of two unsigned numbers
fn euclid_gcd_unsigned<T>(a: T, b: T) -> Result<T, ArrowError>
where
    T: ArrowNativeTypeOp + RemAssign + CheckedNeg,
{
    let (mut a, mut b) = if a > b { (a, b) } else { (b, a) };

    while b != T::ZERO {
        swap(&mut a, &mut b);
        b %= a;
    }

    Ok(a)
}

/// gcd of two unsigned integers
fn euclid_gcd_with_unsigned(a: i64, b: i64) -> Result<i64, ArrowError> {
    let au = a.unsigned_abs();
    let bu = b.unsigned_abs();

    let r = euclid_gcd_unsigned::<u64>(au, bu)?;
    // gcd(i64::MIN, i64::MIN) = i64::MIN.unsigned_abs() cannot fit into i64
    r.try_into().map_err(|_| {
        ArrowError::ComputeError(format!("Signed integer overflow in GCD({a}, {b})"))
    })
}

/// Computes gcd of two unsigned integers using Binary GCD algorithm.
/// Deprecated
pub(super) fn unsigned_gcd(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }

    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            return a << shift;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int64Array};
    use arrow::datatypes::Field;
    use arrow_buffer::i256;
    use datafusion_common::cast::{as_decimal128_array, as_int64_array};
    use datafusion_common::config::ConfigOptions;
    use datafusion_common::ScalarValue;
    use std::sync::Arc;

    #[cfg(test)]
    #[ctor::ctor]
    fn init() {
        // Enable RUST_LOG logging configuration for test
        let _ = env_logger::try_init();
    }

    #[test]
    fn test_i64_array() {
        let arg_fields = vec![
            Field::new("a", DataType::Int64, true).into(),
            Field::new("b", DataType::Int64, true).into(),
        ];
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::new(Int64Array::from(vec![
                    0, 2, 0, 2, 15, 20,
                ]))),
                ColumnarValue::Array(Arc::new(Int64Array::from(vec![
                    0, 0, 2, 3, 10, 1000,
                ]))),
            ],
            arg_fields,
            number_rows: 6,
            return_field: Field::new("f", DataType::Int64, true).into(),
            config_options: Arc::new(ConfigOptions::default()),
        };
        let result = GcdFunc::new()
            .invoke_with_args(args)
            .expect("failed to initialize function");

        match result {
            ColumnarValue::Array(arr) => {
                let values =
                    as_int64_array(&arr).expect("failed to convert result to an array");
                assert_eq!(values.len(), 6);
                assert_eq!(values.value(0), 0);
                assert_eq!(values.value(1), 2);
                assert_eq!(values.value(2), 2);
                assert_eq!(values.value(3), 1);
                assert_eq!(values.value(4), 5);
                assert_eq!(values.value(5), 20);
            }
            ColumnarValue::Scalar(_) => {
                panic!("Expected an array value")
            }
        }
    }

    #[test]
    fn test_decimal_scalar() {
        let arg_fields = vec![
            Field::new("a", DataType::Decimal128(32, 0), true).into(),
            Field::new("a", DataType::Decimal128(32, 0), true).into(),
        ];
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Scalar(ScalarValue::Decimal128(
                    Some(i128::from(15)),
                    32,
                    0,
                )),
                ColumnarValue::Scalar(ScalarValue::Decimal128(
                    Some(i128::from(10)),
                    32,
                    0,
                )),
            ],
            arg_fields,
            number_rows: 1,
            return_field: Field::new("f", DataType::Decimal128(32, 0), true).into(),
            config_options: Arc::new(ConfigOptions::default()),
        };
        let result = GcdFunc::new()
            .invoke_with_args(args)
            .expect("failed to initialize function power");

        match result {
            ColumnarValue::Array(arr) => {
                let ints = as_decimal128_array(&arr)
                    .expect("failed to convert result to an array");

                assert_eq!(ints.len(), 1);
                assert_eq!(ints.value(0), i128::from(5));
                // Signature stays the same as input
                assert_eq!(*arr.data_type(), DataType::Decimal128(32, 0));
            }
            ColumnarValue::Scalar(_) => {
                panic!("Expected an array value")
            }
        }
    }

    const COMMON_TEST_CASES: [(i64, i64, i64); 18] = [
        // Basic cases
        (48, 18, 6),
        (54, 24, 6),
        (100, 50, 50),
        (17, 19, 1),
        (21, 14, 7),
        // Edge cases with 0
        (0, 0, 0),
        (0, 5, 5),
        (10, 0, 10),
        // Same numbers
        (7, 7, 7),
        (100, 100, 100),
        // One is 1
        (1, 1, 1),
        (1, 100, 1),
        (999, 1, 1),
        // Large numbers
        (1000000, 500000, 500000),
        (123456, 789012, 12),
        (999999, 111111, 111111),
        // Powers of 2
        (64, 128, 64),
        (1024, 2048, 1024),
    ];

    #[test]
    fn test_euclid_gcd_i64() {
        let test_cases: Vec<(i64, i64, i64)> = [
            COMMON_TEST_CASES.into(),
            vec![
                // Max value cases
                (1, i64::MAX, 1),
                (i64::MAX, 1, 1),
                (i64::MAX, i64::MAX, i64::MAX),
            ],
        ]
        .concat();

        // Success cases
        for (a, b, expected) in test_cases {
            let actual = euclid_gcd(a, b).expect("should succeed");
            assert_eq!(
                actual, expected,
                "euclid_gcd({}, {}) expected {}, actual {}",
                a, b, expected, actual
            );
        }
    }

    #[test]
    fn test_euclid_gcd_decimal128() {
        let test_cases: Vec<(i256, i256, i256)> = [
            COMMON_TEST_CASES
                .iter()
                .map(|&(a, b, c)| (i256::from(a), i256::from(b), i256::from(c)))
                .collect(),
            vec![
                (i256::from(1), i256::MAX, i256::from(1)),
                (i256::MAX, i256::from(1), i256::from(1)),
                (i256::MAX, i256::MAX, i256::MAX),
            ],
        ]
        .concat();

        // Success cases
        for (a, b, expected) in test_cases {
            let actual = euclid_gcd(a, b).expect("should succeed");
            assert_eq!(
                actual, expected,
                "euclid_gcd({}, {}) expected {}, actual {}",
                a, b, expected, actual
            );
        }
    }
}
