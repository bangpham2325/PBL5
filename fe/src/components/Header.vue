<template>
    <header>
        <div class="left">
            <img src="../assets/logo.svg" alt="TP" class="logo" />
        </div>
        <div class="center">
            <div class="title">
                <h1>Traffic AI</h1>
            </div>
        </div>
        <div class="right">
            <label class="search-bar">
                <img src="../assets/baseline_search_black_24dp.png" alt="Search" class="icon" />
                <input type="text" id="search-bar" name="search-bar" placeholder="Type to search..." />
            </label>
        </div>
    </header>
</template>

<script>
    export default {
        
    }
</script>

<style lang="scss" scoped>
header {
    display: flex;
    justify-content: center;
    padding: 50px 50px 0px;
    font-family: 'Roboto', sans-serif;
    .left {
        flex: 1;
        display: flex;
        justify-content: flex-start;
        .logo {
            width: 55px;
        }
    }
    .center {
        h3 {
            font-size: 32px;
            font-style: italic;
            font-weight: 400;
            margin: 0;
            &.bottom {
                text-align: center;
            }
        }
        h1 {
            color: #171717;
            font-size: 60px;
            margin: 0;
        }
    }
    .right {
        flex: 1;
        display: flex;
        justify-content: flex-end;

        label {
            position: relative;
            display: block;
            width: 300px;
            input {
                width: 100%;
                height: 35px;
                background-color: #FFFFFF;
                box-shadow: 3px 3px 8px 0px rgba(0,0,0,0.2);
                margin: 0;
                padding: 5px 0px 0px 35px;
                box-sizing: border-box;
                appearance: none;
                border: none;
                outline: none;
                color: #171717;
                font-size: 16px;
                align-content: center;
                &::placeholder {
                    color: #888888;
                }
            }
            .icon {
                position: absolute;
                width: 25px;
                margin: 5px;
            }
        }
    }
}
</style>