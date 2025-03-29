''')

    with open(filename, 'w') as f:
        f.write(content)


def main():
    """Main function to generate all program files"""
    dir_name = create_directory()

    # List of generator functions
    generators = [
        generate_program_1,
        generate_program_2,
        generate_program_3,
        generate_program_4,
        generate_program_5,
        generate_program_6,
        generate_program_7,
        generate_program_8,
        generate_program_9,
        generate_program_10,
        generate_program_11,
        generate_program_12,
        generate_program_13,
        generate_program_14,
        generate_program_15,
        generate_program_16,
        generate_program_17,
        generate_program_18,
        generate_program_19,
        generate_program_20
    ]

    # Generate all programs
    for i, generator in enumerate(generators, 1):
        print(f"Generating program {i}...")
        generator(dir_name)

       print(f"\nSuccessfully generated 20 different Python programs in the '{dir_name}' directory.")

if __name__ == "__main__":
    main()
