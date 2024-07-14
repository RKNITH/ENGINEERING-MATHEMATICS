

function createMatrixInputs(matrixId, rowsId, colsId) {
    const rows = document.getElementById(rowsId).value;
    const cols = document.getElementById(colsId).value;
    const matrixInputs = document.getElementById(matrixId + 'Inputs');

    matrixInputs.innerHTML = '';

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            matrixInputs.innerHTML += `<input type="number" id="${matrixId}_${i}_${j}" style="width: 50px;"> `;
        }
        matrixInputs.innerHTML += '<br>';
    }
}

function calculateMultiplication() {
    const rows1 = document.getElementById('rows1').value;
    const cols1 = document.getElementById('cols1').value;
    const rows2 = document.getElementById('rows2').value;
    const cols2 = document.getElementById('cols2').value;

    if (cols1 != rows2) {
        document.getElementById('result').innerText = 'Matrix multiplication is not possible with these dimensions.';
        return;
    }

    const matrix1 = [];
    const matrix2 = [];

    for (let i = 0; i < rows1; i++) {
        matrix1[i] = [];
        for (let j = 0; j < cols1; j++) {
            matrix1[i][j] = parseFloat(document.getElementById(`matrix1_${i}_${j}`).value);
        }
    }

    for (let i = 0; i < rows2; i++) {
        matrix2[i] = [];
        for (let j = 0; j < cols2; j++) {
            matrix2[i][j] = parseFloat(document.getElementById(`matrix2_${i}_${j}`).value);
        }
    }

    const resultMatrix = [];
    for (let i = 0; i < rows1; i++) {
        resultMatrix[i] = [];
        for (let j = 0; j < cols2; j++) {
            resultMatrix[i][j] = 0;
            for (let k = 0; k < cols1; k++) {
                resultMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    let resultString = '';
    for (let i = 0; i < resultMatrix.length; i++) {
        resultString += '[' + resultMatrix[i].join(' ') + ']\n';
    }

    document.getElementById('result').innerText = resultString;
}


function calculateTranspose() {
    const matrixInput = document.getElementById('transposeMatrix').value;
    const matrix = matrixInput.split(';').map(row => row.split(',').map(Number));

    const rows = matrix.length;
    const cols = matrix[0].length;

    const transpose = [];
    for (let i = 0; i < cols; i++) {
        transpose[i] = [];
        for (let j = 0; j < rows; j++) {
            transpose[i][j] = matrix[j][i];
        }
    }

    let resultString = '';
    for (let i = 0; i < transpose.length; i++) {
        resultString += '[' + transpose[i].join(' ') + ']\n';
    }

    document.getElementById('transposeResult').innerText = resultString;
}

function calculateDeterminant() {
    const matrixInput = document.getElementById('detMatrix').value;
    const matrix = matrixInput.split(';').map(row => row.split(',').map(Number));

    const n = matrix.length;
    for (let i = 0; i < n; i++) {
        if (matrix[i].length !== n) {
            document.getElementById('detResult').innerText = 'Please enter a valid square matrix.';
            return;
        }
    }

    function determinant(matrix) {
        if (matrix.length === 1) return matrix[0][0];
        if (matrix.length === 2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

        let det = 0;
        for (let i = 0; i < matrix.length; i++) {
            const subMatrix = matrix.slice(1).map(row => row.filter((_, j) => j !== i));
            det += ((i % 2 === 0 ? 1 : -1) * matrix[0][i] * determinant(subMatrix));
        }
        return det;
    }

    const result = determinant(matrix);
    document.getElementById('detResult').innerText = `Determinant: ${result}`;
}

function calculateInverse() {
    const matrixInput = document.getElementById('inverseMatrix').value;
    const matrix = matrixInput.split(';').map(row => row.split(',').map(Number));

    const n = matrix.length;
    for (let i = 0; i < n; i++) {
        if (matrix[i].length !== n) {
            document.getElementById('inverseResult').innerText = 'Please enter a valid square matrix.';
            return;
        }
    }

    function getMatrixInverse(matrix) {
        const size = matrix.length;
        const identityMatrix = Array.from({ length: size }, (_, i) => Array.from({ length: size }, (_, j) => (i === j ? 1 : 0)));

        for (let i = 0; i < size; i++) {
            let diagElement = matrix[i][i];
            if (diagElement === 0) {
                for (let k = i + 1; k < size; k++) {
                    if (matrix[k][i] !== 0) {
                        [matrix[i], matrix[k]] = [matrix[k], matrix[i]];
                        [identityMatrix[i], identityMatrix[k]] = [identityMatrix[k], identityMatrix[i]];
                        diagElement = matrix[i][i];
                        break;
                    }
                }
            }

            if (diagElement === 0) {
                return null;
            }

            for (let j = 0; j < size; j++) {
                matrix[i][j] /= diagElement;
                identityMatrix[i][j] /= diagElement;
            }

            for (let k = 0; k < size; k++) {
                if (k !== i) {
                    const factor = matrix[k][i];
                    for (let j = 0; j < size; j++) {
                        matrix[k][j] -= factor * matrix[i][j];
                        identityMatrix[k][j] -= factor * identityMatrix[i][j];
                    }
                }
            }
        }

        return identityMatrix;
    }

    const result = getMatrixInverse(matrix);
    if (result === null) {
        document.getElementById('inverseResult').innerText = 'The matrix is singular and does not have an inverse.';
    } else {
        let resultString = '';
        for (let i = 0; i < result.length; i++) {
            resultString += '[' + result[i].join(' ') + ']\n';
        }
        document.getElementById('inverseResult').innerText = resultString;
    }
}

function calculateCramersRule() {
    const matrixInput = document.getElementById('cramerMatrix').value;
    const constantsInput = document.getElementById('cramerConstants').value;
    const matrix = matrixInput.split(';').map(row => row.split(',').map(Number));
    const constants = constantsInput.split(',').map(Number);

    const n = matrix.length;
    for (let i = 0; i < n; i++) {
        if (matrix[i].length !== n) {
            document.getElementById('cramerResult').innerText = 'Please enter a valid square matrix.';
            return;
        }
    }

    function determinant(matrix) {
        if (matrix.length === 1) return matrix[0][0];
        if (matrix.length === 2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

        let det = 0;
        for (let i = 0; i < matrix.length; i++) {
            const subMatrix = matrix.slice(1).map(row => row.filter((_, j) => j !== i));
            det += ((i % 2 === 0 ? 1 : -1) * matrix[0][i] * determinant(subMatrix));
        }
        return det;
    }

    const detA = determinant(matrix);
    if (detA === 0) {
        document.getElementById('cramerResult').innerText = 'The determinant is zero; no unique solution exists.';
        return;
    }

    const results = [];
    for (let i = 0; i < n; i++) {
        const Ai = matrix.map(row => row.slice());
        for (let j = 0; j < n; j++) {
            Ai[j][i] = constants[j];
        }
        results.push(determinant(Ai) / detA);
    }

    document.getElementById('cramerResult').innerText = `Solutions: ${results.join(', ')}`;
}

function checkConsistency() {
    const matrixInput = document.getElementById('consistencyMatrix').value;
    const constantsInput = document.getElementById('consistencyConstants').value;
    const matrix = matrixInput.split(';').map(row => row.split(',').map(Number));
    const constants = constantsInput.split(',').map(Number);

    const augmentedMatrix = matrix.map((row, i) => [...row, constants[i]]);
    const rowEchelon = rowEchelonForm(augmentedMatrix);

    let consistent = true;
    for (let i = 0; i < rowEchelon.length; i++) {
        if (rowEchelon[i].slice(0, -1).every(v => v === 0) && rowEchelon[i][rowEchelon[i].length - 1] !== 0) {
            consistent = false;
            break;
        }
    }

    document.getElementById('consistencyResult').innerText = consistent ? 'The system is consistent.' : 'The system is inconsistent.';
}

function rowEchelonForm(matrix) {
    const m = matrix.length, n = matrix[0].length;
    for (let i = 0; i < m; i++) {
        // Find pivot
        let pivot = i;
        while (pivot < m && matrix[pivot][i] === 0) {
            pivot++;
        }
        if (pivot === m) continue; // No pivot found

        // Swap rows
        if (pivot !== i) {
            [matrix[i], matrix[pivot]] = [matrix[pivot], matrix[i]];
        }

        // Eliminate below
        for (let j = i + 1; j < m; j++) {
            const factor = matrix[j][i] / matrix[i][i];
            for (let k = i; k < n; k++) {
                matrix[j][k] -= factor * matrix[i][k];
            }
        }
    }
    return matrix;
}

// RANK CALCULATION

function parseMatrix(matrixString) {
    return matrixString.trim().split(';').map(row => row.split(',').map(Number));
}

function getRowEchelonForm(matrix) {
    let lead = 0;
    const rowCount = matrix.length;
    const columnCount = matrix[0].length;

    for (let r = 0; r < rowCount; r++) {
        if (lead >= columnCount) {
            return;
        }
        let i = r;
        while (matrix[i][lead] === 0) {
            i++;
            if (i === rowCount) {
                i = r;
                lead++;
                if (lead === columnCount) {
                    return;
                }
            }
        }
        [matrix[i], matrix[r]] = [matrix[r], matrix[i]];
        const lv = matrix[r][lead];
        for (let j = 0; j < columnCount; j++) {
            matrix[r][j] /= lv;
        }
        for (let i = 0; i < rowCount; i++) {
            if (i !== r) {
                const lv = matrix[i][lead];
                for (let j = 0; j < columnCount; j++) {
                    matrix[i][j] -= lv * matrix[r][j];
                }
            }
        }
        lead++;
    }
}

function calculateRank() {
    const matrixString = document.getElementById('rankMatrix').value;
    const matrix = parseMatrix(matrixString);

    getRowEchelonForm(matrix);

    let rank = 0;
    for (let i = 0; i < matrix.length; i++) {
        if (matrix[i].some(value => value !== 0)) {
            rank++;
        }
    }

    document.getElementById('rankResult').innerText = `Rank of the matrix is: ${rank}`;
}

// NORMAL FORM
function parseMatrix(input) {
    return input.split(';').map(row => row.split(',').map(Number));
}

function createMatrixTable(matrix) {
    let table = '<table border="1" style="border-collapse: collapse; text-align: center;">';
    matrix.forEach(row => {
        table += '<tr>';
        row.forEach(cell => {
            table += `<td>${cell}</td>`;
        });
        table += '</tr>';
    });
    table += '</table>';
    return table;
}

function cloneMatrix(matrix) {
    return matrix.map(row => [...row]);
}

function calculateNormalForm() {
    const input = document.getElementById('normalFormMatrix').value;
    const matrix = parseMatrix(input);
    const steps = [];

    function addStep(matrix, stepDescription) {
        steps.push(stepDescription + '<br>' + createMatrixTable(cloneMatrix(matrix)));
    }

    let m = cloneMatrix(matrix);
    let rows = m.length;
    let cols = m[0].length;

    for (let i = 0; i < rows; i++) {
        let diagElement = m[i][i];
        for (let j = 0; j < cols; j++) {
            m[i][j] = m[i][j] / diagElement;
        }
        addStep(m, `Normalize row ${i + 1}`);

        for (let k = i + 1; k < rows; k++) {
            let factor = m[k][i];
            for (let j = 0; j < cols; j++) {
                m[k][j] = m[k][j] - factor * m[i][j];
            }
            addStep(m, `Eliminate row ${k + 1} column ${i + 1}`);
        }
    }

    for (let i = rows - 1; i >= 0; i--) {
        for (let k = i - 1; k >= 0; k--) {
            let factor = m[k][i];
            for (let j = 0; j < cols; j++) {
                m[k][j] = m[k][j] - factor * m[i][j];
            }
            addStep(m, `Eliminate row ${k + 1} column ${i + 1}`);
        }
    }

    const resultElement = document.getElementById('normalFormResult');
    const stepsElement = document.getElementById('normalFormSteps');
    resultElement.innerHTML = `<strong>Normal Form:</strong><br>${createMatrixTable(m)}`;
    stepsElement.innerHTML = `<strong>Steps:</strong><br>${steps.join('<br><br>')}`;
}




// EIGEN VALUE

function parseMatrixInput(inputId) {
    const matrixText = document.getElementById(inputId).value;
    const rows = matrixText.trim().split(';');
    return rows.map(row => row.trim().split(',').map(Number));
}

function calculateEigen() {
    const matrix = parseMatrixInput('eigenMatrix');
    if (matrix.length !== matrix[0].length) {
        document.getElementById('eigenResult').textContent = 'Please enter a square matrix.';
        return;
    }

    const n = matrix.length;
    if (n === 3) {
        calculateEigen3x3(matrix);
    } else {
        document.getElementById('eigenResult').textContent = 'Currently, only 3x3 matrices are supported.';
    }
}

function calculateEigen3x3(matrix) {
    const a = matrix[0][0];
    const b = matrix[0][1];
    const c = matrix[0][2];
    const d = matrix[1][0];
    const e = matrix[1][1];
    const f = matrix[1][2];
    const g = matrix[2][0];
    const h = matrix[2][1];
    const i = matrix[2][2];

    // Calculate the characteristic polynomial coefficients
    const A = -1;
    const B = a + e + i;
    const C = a * e + a * i + e * i - c * g - b * d - f * h;
    const D = a * e * i + b * f * g + c * d * h - a * f * h - b * d * i - c * e * g;

    const steps = [];
    steps.push(`Step 1: The input matrix is A = [ [${a}, ${b}, ${c}], [${d}, ${e}, ${f}], [${g}, ${h}, ${i}] ]`);
    steps.push(`Step 2: The characteristic polynomial is |A - λI| = λ³ - (${B})λ² + (${C})λ - ${D} = 0`);
    steps.push(`Step 3: Solve the cubic equation to find the eigenvalues`);

    const eigenvalues = solveCubic(A, B, C, D);

    if (eigenvalues.length === 0) {
        steps.push('No real eigenvalues found.');
        document.getElementById('eigenResult').textContent = 'No real eigenvalues found.';
        document.getElementById('eigenSteps').innerHTML = steps.join('<br>');
        return;
    }

    const eigenvectors = eigenvalues.map(lambda => findEigenvector(matrix, lambda));

    steps.push(`Step 4: Eigenvalues are λ = ${eigenvalues.join(', ')}`);
    steps.push(`Step 5: Calculate the eigenvectors for each eigenvalue`);

    eigenvectors.forEach((vector, index) => {
        steps.push(`Eigenvector for λ = ${eigenvalues[index]}: [${vector.map(v => v.toFixed(2)).join(', ')}]`);
    });

    document.getElementById('eigenResult').textContent = `Eigenvalues: ${eigenvalues.join(', ')}`;
    document.getElementById('eigenResult').textContent += `\nEigenvectors: ${eigenvectors.map(v => '[' + v.map(e => e.toFixed(2)).join(', ') + ']').join(', ')}`;

    document.getElementById('eigenSteps').innerHTML = steps.join('<br>');
}

function solveCubic(a, b, c, d) {
    // Cardano's formula for solving cubic equations
    const f = ((3 * c / a) - ((b * b) / (a * a))) / 3;
    const g = ((2 * b * b * b) / (a * a * a) - (9 * b * c) / (a * a) + (27 * d / a)) / 27;
    const h = (g * g / 4) + (f * f * f / 27);

    if (h > 0) {
        const r = -(g / 2) + Math.sqrt(h);
        const s = Math.cbrt(r);
        const t = -(g / 2) - Math.sqrt(h);
        const u = Math.cbrt(t);
        const root1 = (s + u) - (b / (3 * a));
        return [root1];
    } else {
        const i = Math.sqrt(((g * g / 4) - h));
        const j = Math.cbrt(i);
        const k = Math.acos(-(g / (2 * i)));
        const l = -j;
        const m = Math.cos(k / 3);
        const n = Math.sqrt(3) * Math.sin(k / 3);
        const p = -(b / (3 * a));
        const root1 = 2 * j * Math.cos(k / 3) - (b / (3 * a));
        const root2 = l * (m + n) + p;
        const root3 = l * (m - n) + p;
        return [root1, root2, root3];
    }
}

function findEigenvector(matrix, lambda) {
    const n = matrix.length;
    const augmentedMatrix = matrix.map((row, i) => {
        const newRow = row.slice();
        newRow[i] -= lambda;
        return newRow;
    });

    // Row reduce the augmented matrix
    for (let i = 0; i < n; i++) {
        // Make the diagonal contain all 1's
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
            if (Math.abs(augmentedMatrix[k][i]) > Math.abs(augmentedMatrix[maxRow][i])) {
                maxRow = k;
            }
        }
        [augmentedMatrix[i], augmentedMatrix[maxRow]] = [augmentedMatrix[maxRow], augmentedMatrix[i]];

        for (let k = i + 1; k < n; k++) {
            const factor = augmentedMatrix[k][i] / augmentedMatrix[i][i];
            for (let j = i; j < n; j++) {
                augmentedMatrix[k][j] -= factor * augmentedMatrix[i][j];
            }
        }
    }

    const eigenvector = new Array(n).fill(1);
    for (let i = n - 1; i >= 0; i--) {
        let sum = 0;
        for (let j = i + 1; j < n; j++) {
            sum += augmentedMatrix[i][j] * eigenvector[j];
        }
        eigenvector[i] = (1 - sum) / augmentedMatrix[i][i];
    }

    return eigenvector;
}


// diagonal form


function parseMatrixInput(inputId) {
    const matrixText = document.getElementById(inputId).value;
    const rows = matrixText.trim().split(';');
    return rows.map(row => row.trim().split(',').map(Number));
}

function calculateDiagonalForm() {
    const matrix = parseMatrixInput('diagonalMatrix');
    if (matrix.length !== matrix[0].length) {
        document.getElementById('diagonalFormResult').textContent = 'Please enter a square matrix.';
        return;
    }

    const n = matrix.length;
    if (n === 3) {
        calculateDiagonalForm3x3(matrix);
    } else {
        document.getElementById('diagonalFormResult').textContent = 'Currently, only 3x3 matrices are supported.';
    }
}

function calculateDiagonalForm3x3(matrix) {
    const eigenvalues = calculateEigenvalues3x3(matrix);
    const eigenvectors = eigenvalues.map(lambda => findEigenvector(matrix, lambda));

    const steps = [];
    steps.push(`Step 1: The input matrix is A = [ [${matrix[0].join(', ')}], [${matrix[1].join(', ')}], [${matrix[2].join(', ')}] ]`);
    steps.push(`Step 2: Calculate the eigenvalues`);
    steps.push(`Eigenvalues: ${eigenvalues.join(', ')}`);

    steps.push(`Step 3: Calculate the eigenvectors`);
    eigenvectors.forEach((vector, index) => {
        steps.push(`Eigenvector for λ = ${eigenvalues[index]}: [${vector.map(v => v.toFixed(2)).join(', ')}]`);
    });

    const P = eigenvectors;
    const P_inv = invertMatrix(P);
    const D = eigenvalues.map((lambda, i) => {
        const row = new Array(eigenvalues.length).fill(0);
        row[i] = lambda;
        return row;
    });

    steps.push(`Step 4: Form the matrix P using eigenvectors as columns`);
    steps.push(`P = [ [${P[0].join(', ')}], [${P[1].join(', ')}], [${P[2].join(', ')}] ]`);

    steps.push(`Step 5: Calculate P^-1 (inverse of P)`);
    steps.push(`P^-1 = [ [${P_inv[0].join(', ')}], [${P_inv[1].join(', ')}], [${P_inv[2].join(', ')}] ]`);

    steps.push(`Step 6: Form the diagonal matrix D using the eigenvalues`);
    steps.push(`D = [ [${D[0].join(', ')}], [${D[1].join(', ')}], [${D[2].join(', ')}] ]`);

    document.getElementById('diagonalFormResult').innerHTML = steps.join('<br>');
}

function calculateEigenvalues3x3(matrix) {
    const a = matrix[0][0];
    const b = matrix[0][1];
    const c = matrix[0][2];
    const d = matrix[1][0];
    const e = matrix[1][1];
    const f = matrix[1][2];
    const g = matrix[2][0];
    const h = matrix[2][1];
    const i = matrix[2][2];

    const A = -1;
    const B = a + e + i;
    const C = a * e + a * i + e * i - c * g - b * d - f * h;
    const D = a * e * i + b * f * g + c * d * h - a * f * h - b * d * i - c * e * g;

    return solveCubic(A, B, C, D);
}

function solveCubic(a, b, c, d) {
    // Cardano's formula for solving cubic equations
    const f = ((3 * c / a) - ((b * b) / (a * a))) / 3;
    const g = ((2 * b * b * b) / (a * a * a) - (9 * b * c) / (a * a) + (27 * d / a)) / 27;
    const h = (g * g / 4) + (f * f * f / 27);

    if (h > 0) {
        const r = -(g / 2) + Math.sqrt(h);
        const s = Math.cbrt(r);
        const t = -(g / 2) - Math.sqrt(h);
        const u = Math.cbrt(t);
        const root1 = (s + u) - (b / (3 * a));
        return [root1];
    } else {
        const i = Math.sqrt(((g * g / 4) - h));
        const j = Math.cbrt(i);
        const k = Math.acos(-(g / (2 * i)));
        const l = -j;
        const m = Math.cos(k / 3);
        const n = Math.sqrt(3) * Math.sin(k / 3);
        const p = -(b / (3 * a));
        const root1 = 2 * j * Math.cos(k / 3) - (b / (3 * a));
        const root2 = l * (m + n) + p;
        const root3 = l * (m - n) + p;
        return [root1, root2, root3];
    }
}

function findEigenvector(matrix, lambda) {
    const n = matrix.length;
    const augmentedMatrix = matrix.map((row, i) => {
        const newRow = row.slice();
        newRow[i] -= lambda;
        return newRow;
    });

    // Row reduce the augmented matrix
    for (let i = 0; i < n; i++) {
        // Make the diagonal contain all 1's
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
            if (Math.abs(augmentedMatrix[k][i]) > Math.abs(augmentedMatrix[maxRow][i])) {
                maxRow = k;
            }
        }
        [augmentedMatrix[i], augmentedMatrix[maxRow]] = [augmentedMatrix[maxRow], augmentedMatrix[i]];

        for (let k = i + 1; k < n; k++) {
            const factor = augmentedMatrix[k][i] / augmentedMatrix[i][i];
            for (let j = i; j < n; j++) {
                augmentedMatrix[k][j] -= factor * augmentedMatrix[i][j];
            }
        }
    }

    const eigenvector = new Array(n).fill(1);
    for (let i = n - 1; i >= 0; i--) {
        let sum = 0;
        for (let j = i + 1; j < n; j++) {
            sum += augmentedMatrix[i][j] * eigenvector[j];
        }
        eigenvector[i] = (1 - sum) / augmentedMatrix[i][i];
    }

    return eigenvector;
}

function invertMatrix(matrix) {
    const n = matrix.length;
    const identityMatrix = matrix.map((row, i) => row.map((_, j) => (i === j ? 1 : 0)));
    const augmentedMatrix = matrix.map((row, i) => row.concat(identityMatrix[i]));

    // Row reduce the augmented matrix
    for (let i = 0; i < n; i++) {
        // Make the diagonal contain all 1's
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
            if (Math.abs(augmentedMatrix[k][i]) > Math.abs(augmentedMatrix[maxRow][i])) {
                maxRow = k;
            }
        }
        [augmentedMatrix[i], augmentedMatrix[maxRow]] = [augmentedMatrix[maxRow], augmentedMatrix[i]];

        const diagValue = augmentedMatrix[i][i];
        for (let j = 0; j < 2 * n; j++) {
            augmentedMatrix[i][j] /= diagValue;
        }

        for (let k = 0; k < n; k++) {
            if (k !== i) {
                const factor = augmentedMatrix[k][i];
                for (let j = 0; j < 2 * n; j++) {
                    augmentedMatrix[k][j] -= factor * augmentedMatrix[i][j];
                }
            }
        }
    }

    const inverseMatrix = augmentedMatrix.map(row => row.slice(n));
    return inverseMatrix;
}



