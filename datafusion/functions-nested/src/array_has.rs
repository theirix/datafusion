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

//! [`ScalarUDFImpl`] definitions for array_has, array_has_all and array_has_any functions.

use arrow::array::{Array, ArrayRef, BooleanArray, OffsetSizeTrait};
use arrow::datatypes::DataType;
use arrow::row::{RowConverter, Rows, SortField};
use arrow_array::GenericListArray;
use datafusion_common::cast::as_generic_list_array;
use datafusion_common::utils::string_utils::string_array_to_vec;
use datafusion_common::{exec_err, Result};
use datafusion_expr::{ColumnarValue, ScalarUDFImpl, Signature, Volatility};

use itertools::Itertools;

use crate::utils::make_scalar_function;

use std::any::Any;
use std::sync::Arc;

// Create static instances of ScalarUDFs for each function
make_udf_expr_and_func!(ArrayHas,
    array_has,
    haystack_array element, // arg names
    "returns true, if the element appears in the first array, otherwise false.", // doc
    array_has_udf // internal function name
);
make_udf_expr_and_func!(ArrayHasAll,
    array_has_all,
    haystack_array needle_array, // arg names
    "returns true if each element of the second array appears in the first array; otherwise, it returns false.", // doc
    array_has_all_udf // internal function name
);
make_udf_expr_and_func!(ArrayHasAny,
    array_has_any,
    haystack_array needle_array, // arg names
    "returns true if at least one element of the second array appears in the first array; otherwise, it returns false.", // doc
    array_has_any_udf // internal function name
);

#[derive(Debug)]
pub struct ArrayHas {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for ArrayHas {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayHas {
    pub fn new() -> Self {
        Self {
            signature: Signature::array_and_element(Volatility::Immutable),
            aliases: vec![
                String::from("list_has"),
                String::from("array_contains"),
                String::from("list_contains"),
            ],
        }
    }
}

impl ScalarUDFImpl for ArrayHas {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "array_has"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        make_scalar_function(array_has_inner)(args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }
}

fn array_has_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    match args[0].data_type() {
        DataType::List(_) => array_has_dispatch::<i32>(&args[0], &args[1]),
        DataType::LargeList(_) => array_has_dispatch::<i64>(&args[0], &args[1]),
        _ => exec_err!(
            "array_has does not support type '{:?}'.",
            args[0].data_type()
        ),
    }
}

fn array_has_all_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    match args[0].data_type() {
        DataType::List(_) => {
            array_has_all_and_any_dispatch::<i32>(&args[0], &args[1], ComparisonType::All)
        }
        DataType::LargeList(_) => {
            array_has_all_and_any_dispatch::<i64>(&args[0], &args[1], ComparisonType::All)
        }
        _ => exec_err!(
            "array_has does not support type '{:?}'.",
            args[0].data_type()
        ),
    }
}

fn array_has_any_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    match args[0].data_type() {
        DataType::List(_) => {
            array_has_all_and_any_dispatch::<i32>(&args[0], &args[1], ComparisonType::Any)
        }
        DataType::LargeList(_) => {
            array_has_all_and_any_dispatch::<i64>(&args[0], &args[1], ComparisonType::Any)
        }
        _ => exec_err!(
            "array_has does not support type '{:?}'.",
            args[0].data_type()
        ),
    }
}

#[derive(Debug)]
pub struct ArrayHasAll {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for ArrayHasAll {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayHasAll {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
            aliases: vec![String::from("list_has_all")],
        }
    }
}

impl ScalarUDFImpl for ArrayHasAll {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "array_has_all"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        make_scalar_function(array_has_all_inner)(args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }
}

#[derive(Debug)]
pub struct ArrayHasAny {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for ArrayHasAny {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayHasAny {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
            aliases: vec![String::from("list_has_any")],
        }
    }
}

impl ScalarUDFImpl for ArrayHasAny {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "array_has_any"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        make_scalar_function(array_has_any_inner)(args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }
}

/// Represents the type of comparison for array_has.
#[derive(Debug, PartialEq, Clone, Copy)]
enum ComparisonType {
    // array_has_all
    All,
    // array_has_any
    Any,
}

fn array_has_dispatch<O: OffsetSizeTrait>(
    haystack: &ArrayRef,
    needle: &ArrayRef,
) -> Result<ArrayRef> {
    let haystack = as_generic_list_array::<O>(haystack)?;
    match needle.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            array_has_string_internal::<O>(haystack, needle)
        }
        _ => general_array_has::<O>(haystack, needle),
    }
}

fn array_has_all_and_any_dispatch<O: OffsetSizeTrait>(
    haystack: &ArrayRef,
    needle: &ArrayRef,
    comparison_type: ComparisonType,
) -> Result<ArrayRef> {
    let haystack = as_generic_list_array::<O>(haystack)?;
    let needle = as_generic_list_array::<O>(needle)?;
    match needle.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            array_has_all_and_any_string_internal::<O>(haystack, needle, comparison_type)
        }
        _ => general_array_has_for_all_and_any::<O>(haystack, needle, comparison_type),
    }
}

fn array_has_string_internal<O: OffsetSizeTrait>(
    haystack: &GenericListArray<O>,
    needle: &ArrayRef,
) -> Result<ArrayRef> {
    let mut boolean_builder = BooleanArray::builder(haystack.len());
    for (arr, element) in haystack.iter().zip(string_array_to_vec(needle).into_iter()) {
        match (arr, element) {
            (Some(arr), Some(element)) => {
                boolean_builder.append_value(
                    string_array_to_vec(&arr)
                        .into_iter()
                        .flatten()
                        .any(|x| x == element),
                );
            }
            (_, _) => {
                boolean_builder.append_null();
            }
        }
    }

    Ok(Arc::new(boolean_builder.finish()))
}

fn general_array_has<O: OffsetSizeTrait>(
    array: &GenericListArray<O>,
    needle: &ArrayRef,
) -> Result<ArrayRef> {
    let mut boolean_builder = BooleanArray::builder(array.len());
    let converter = RowConverter::new(vec![SortField::new(array.value_type())])?;
    let sub_arr_values = converter.convert_columns(&[Arc::clone(needle)])?;

    for (row_idx, arr) in array.iter().enumerate() {
        if let Some(arr) = arr {
            let arr_values = converter.convert_columns(&[arr])?;
            boolean_builder.append_value(
                arr_values
                    .iter()
                    .dedup()
                    .any(|x| x == sub_arr_values.row(row_idx)),
            );
        } else {
            boolean_builder.append_null();
        }
    }

    Ok(Arc::new(boolean_builder.finish()))
}

// String comparison for array_has_all and array_has_any
fn array_has_all_and_any_string_internal<O: OffsetSizeTrait>(
    array: &GenericListArray<O>,
    needle: &GenericListArray<O>,
    comparison_type: ComparisonType,
) -> Result<ArrayRef> {
    let mut boolean_builder = BooleanArray::builder(array.len());
    for (arr, sub_arr) in array.iter().zip(needle.iter()) {
        match (arr, sub_arr) {
            (Some(arr), Some(sub_arr)) => {
                let haystack_array = string_array_to_vec(&arr);
                let needle_array = string_array_to_vec(&sub_arr);
                boolean_builder.append_value(array_has_string_kernel(
                    haystack_array,
                    needle_array,
                    comparison_type,
                ));
            }
            (_, _) => {
                boolean_builder.append_null();
            }
        }
    }

    Ok(Arc::new(boolean_builder.finish()))
}

fn array_has_string_kernel(
    haystack: Vec<Option<&str>>,
    needle: Vec<Option<&str>>,
    comparison_type: ComparisonType,
) -> bool {
    match comparison_type {
        ComparisonType::All => needle
            .iter()
            .dedup()
            .all(|x| haystack.iter().dedup().any(|y| y == x)),
        ComparisonType::Any => needle
            .iter()
            .dedup()
            .any(|x| haystack.iter().dedup().any(|y| y == x)),
    }
}

// General row comparison for array_has_all and array_has_any
fn general_array_has_for_all_and_any<O: OffsetSizeTrait>(
    haystack: &GenericListArray<O>,
    needle: &GenericListArray<O>,
    comparison_type: ComparisonType,
) -> Result<ArrayRef> {
    let mut boolean_builder = BooleanArray::builder(haystack.len());
    let converter = RowConverter::new(vec![SortField::new(haystack.value_type())])?;

    for (arr, sub_arr) in haystack.iter().zip(needle.iter()) {
        if let (Some(arr), Some(sub_arr)) = (arr, sub_arr) {
            let arr_values = converter.convert_columns(&[arr])?;
            let sub_arr_values = converter.convert_columns(&[sub_arr])?;
            boolean_builder.append_value(general_array_has_all_and_any_kernel(
                arr_values,
                sub_arr_values,
                comparison_type,
            ));
        } else {
            boolean_builder.append_null();
        }
    }

    Ok(Arc::new(boolean_builder.finish()))
}

fn general_array_has_all_and_any_kernel(
    haystack_rows: Rows,
    needle_rows: Rows,
    comparison_type: ComparisonType,
) -> bool {
    match comparison_type {
        ComparisonType::All => needle_rows.iter().all(|needle_row| {
            haystack_rows
                .iter()
                .any(|haystack_row| haystack_row == needle_row)
        }),
        ComparisonType::Any => needle_rows.iter().any(|needle_row| {
            haystack_rows
                .iter()
                .any(|haystack_row| haystack_row == needle_row)
        }),
    }
}
